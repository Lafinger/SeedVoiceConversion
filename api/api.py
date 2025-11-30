import argparse
import os
import threading
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Optional, Awaitable, Any

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru_settings import TraceID, logger, setup_logging

from music_utils import merge_accompaniment_and_vocals
from voice_conversion_service import ConversionCancelled, VoiceConversionService

BASE_DIR = Path(__file__).resolve().parent
UVICORN_LOG_CONFIG = BASE_DIR / "uvicorn_config.json"

voice_service = VoiceConversionService()

# 使用全局信号量进行限流检测
global_semaphore = anyio.Semaphore(1)

app = FastAPI(
    title="Seed Voice Conversion Service",
    description="提供 Seed-VC 声音转换能力的 HTTP API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，如["https://example.com", "https://app.example.com"]
    allow_credentials=True, # 允许跨域请求携带凭据（Cookie、认证头、TLS客户端证书）
    allow_methods=["*"],  # 允许所有方法， 如：["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
    allow_headers=["*"],  # 允许所有请求头， 如：["Content-Type", "Authorization", "X-Request-Id"]
    expose_headers=["*"],  # 暴露所有请求头， 如：["Content-Type", "Authorization", "X-Request-Id"]
    max_age=600, # 预检请求（OPTIONS）结果的缓存时间（秒）, 减少OPTIONS请求频率，提高性能
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """
    1.设置日志的全链路追踪
    2.记录错误日志
    """
    try:
        REQUEST_ID_KEY = "X-Request-Id"
        _req_id_val = request.headers.get(REQUEST_ID_KEY, "??????") # 如果请求头中没有X-Request-Id，则设置为??????
        req_id = TraceID.set(_req_id_val)
        logger.info(f"{request.method} {request.url}")
        response = await call_next(request)
        response.headers[REQUEST_ID_KEY] = req_id.get()
        return response
    except Exception as ex:
        logger.exception(ex)  # 这个方法能记录错误栈
        return JSONResponse(content={"success": False}, status_code=500)
    finally:
        pass


async def disconnected(request: Request, cancel_event: threading.Event) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            cancel_event.set()
            break


async def _temp_save_upload_file(upload: UploadFile) -> Path:
    """
    将上传的文件保存到临时目录，返回保存后的路径。
    """
    suffix = Path(upload.filename or "").suffix or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                tmp_file.write(chunk)
    finally:
        await upload.close()
    return Path(tmp_path)

@app.post(
    "/api/v1/voice-conversion",
    summary="Seed-VC 声音转换",
    response_class=FileResponse,
)
async def voice_conversion_endpoint(
    request: Request,
    source_audio: UploadFile = File(..., description="待转换的源音频文件"),
    reference_audio: UploadFile = File(..., description="参考音色音频文件"),
    diffusion_steps: Annotated[int, Form(description="扩散步数", ge=1, le=100)] = 10,
    length_adjust: Annotated[float, Form(description="输出长度调节系数", ge=0.5, le=2.0)] = 1.0,
    inference_cfg_rate: Annotated[
        float, Form(description="classifier-free guidance 系数", ge=0.0, le=5.0)
    ] = 0.7,
    auto_f0_adjust: Annotated[bool, Form(description="是否自动对齐音高中位数")] = True,
    pitch_shift: Annotated[int, Form(description="手动升降调，单位半音", ge=-12, le=12)] = 0,
) -> FileResponse:
    """
    提供声音转换能力的路由，返回生成的 wav 文件。
    """
    
    # 检查全局信号量是否被占用
    if global_semaphore.value == 0:
        logger.error("global semaphore locked, request rejected")
        raise HTTPException(status_code=499, detail="连接断开任务已取消")
    else:
        await global_semaphore.acquire()

    logger.info("conversion initialization")
    source_path = await _temp_save_upload_file(source_audio)
    reference_path = await _temp_save_upload_file(reference_audio)
    cancel_event = threading.Event()

    try:
        logger.info("conversion task creation")

        # 用 anyio 的 TaskGroup + CancelScope
        async with anyio.create_task_group() as tg:
            # 开一个“监听断连”的协程，断开时会调用 tg.cancel_scope.cancel()
            tg.start_soon(disconnected, request, cancel_event)

            # 在同一个 cancel_scope 里，把同步的 convert 丢到线程执行
            output_path = await anyio.to_thread.run_sync(
                lambda: voice_service.convert(
                    source_path=source_path,
                    reference_path=reference_path,
                    diffusion_steps=diffusion_steps,
                    length_adjust=length_adjust,
                    inference_cfg_rate=inference_cfg_rate,
                    auto_f0_adjust=auto_f0_adjust,
                    pitch_shift=pitch_shift,
                    cancel_event=cancel_event,
                ),
                cancellable=False,
            )
            tg.cancel_scope.cancel()

        logger.info("conversion completed")

        # 走到这里，说明既没有异常也没有被取消
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )

    # anyio 的取消异常（兼容 Trio/asyncio）
    except ConversionCancelled:
        logger.error("conversion cancelled by client")
        raise HTTPException(status_code=499, detail="连接断开任务已取消")
    except anyio.get_cancelled_exc_class():
        logger.error("client cancel connection")
        raise HTTPException(status_code=499, detail="连接断开任务已取消")


    except AssertionError as e:
        logger.error(e)
        raise HTTPException(status_code=501, detail="参数验证失败")

    except Exception as e:
        logger.error("conversion failed: %s", e)
        raise HTTPException(status_code=500, detail="声音转换失败")

    finally:
        global_semaphore.release()
        cancel_event.clear()

        # 清理临时文件
        for tmp in (source_path, reference_path):
            try:
                tmp.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning("清理临时文件失败 %s: %s", tmp, cleanup_error)

        logger.info("semaphore released after convert completion")


@app.post(
    "/api/v1/mix-accompaniment",
    summary="合并伴奏和人声",
    response_class=FileResponse,
)
async def mix_accompaniment_and_vocals(
    accompaniment_audio: UploadFile = File(..., description="伴奏音频文件"),
    vocal_audio: UploadFile = File(..., description="人声音频文件"),
    accompaniment_gain_db: Annotated[
        float, Form(description="伴奏增益（dB）", ge=-30.0, le=30.0)
    ] = 0.0,
    vocal_gain_db: Annotated[float, Form(description="人声增益（dB）", ge=-30.0, le=30.0)] = 0.0,
) -> FileResponse:
    """
    接收伴奏与人声两条音轨，做基础混音后返回合成的 wav 文件。
    """
    accompaniment_path = await _temp_save_upload_file(accompaniment_audio)
    vocal_path = await _temp_save_upload_file(vocal_audio)

    try:
        output_path = await anyio.to_thread.run_sync(
            lambda: merge_accompaniment_and_vocals(
                accompaniment_path=accompaniment_path,
                vocal_path=vocal_path,
                accompaniment_gain_db=accompaniment_gain_db,
                vocal_gain_db=vocal_gain_db,
            ),
            cancellable=False,
        )
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )
    except Exception as e:
        logger.error("mix accompaniment and vocals failed: %s", e)
        raise HTTPException(status_code=500, detail="合并伴奏与人声失败")
    finally:
        for tmp in (accompaniment_path, vocal_path):
            try:
                tmp.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning("清理临时文件失败 %s: %s", tmp, cleanup_error)

@app.get("/api/v1/health", summary="健康检查")
async def health_check():
    return {"status": "ok"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Voice Conversion HTTP Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=5656, help="服务监听端口")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()

    # 解析命令行参数
    args = parse_arguments()

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=str(UVICORN_LOG_CONFIG),
        log_level="info",
    )
