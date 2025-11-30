# SeedVoiceConversion

Seed Voice Conversion 的 FastAPI 推理服务，封装 Seed-VC 声音转换与伴奏/人声合并能力，提供 HTTP API 及简单客户端脚本。

## 功能特性
- Seed-VC diffusion 声音转换，支持自动/手动音高调整、CFG、长度伸缩
- FastAPI HTTP 服务，Swagger 文档 `/api/v1/docs`，健康检查 `/api/v1/health`
- 伴奏与人声混音接口，支持分贝增益调节
- 日志追踪（request id + loguru + uvicorn），推理结果自动保存到 `outputs/`
- Hugging Face 模型自动下载并缓存到 `checkpoints/hf_cache`

## 项目结构
- `api/api.py`：FastAPI 入口，定义 HTTP 路由
- `api/voice_conversion_service.py`：声音转换服务单例封装
- `api/seed_conversion.py`：核心推理逻辑与模型加载
- `api/music_utils.py`：伴奏/人声合成工具
- `api/loguru_settings.py`：日志配置
- `api/modules/**`：Seed-VC 依赖的模型组件实现
- `client.py`：HTTP 调用示例脚本
- `requirements-win.txt`：Windows 依赖列表
- `checkpoints/`：模型与缓存目录（运行时生成）
- `musics/`、`references/`：示例音频
- `outputs/`：推理结果目录（运行时生成）
- `uvicorn_config.json`：Uvicorn 日志配置

## 环境要求
- Windows 11 / Python 3.10+
- GPU 可选（优先使用 CUDA，有则推理更快），无 GPU 可退化为 CPU
- 推荐安装 FFmpeg 以保证 `pydub` 读写音频正常
- 需要联网以首次下载 Hugging Face 权重；缓存后可离线运行

## 安装与运行
1. 创建并激活虚拟环境（已有 `.venv` 可直接使用）：
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. 安装依赖：
```
.\.venv\Scripts\python.exe -m pip install -r requirements-win.txt
```
3. 启动服务（默认端口 5656，可自定义）：
```
.\.venv\Scripts\python.exe api\api.py --host 0.0.0.0 --port 8000
```
4. 访问文档与接口：
   - Swagger: http://127.0.0.1:8000/api/v1/docs
   - Redoc: http://127.0.0.1:8000/api/v1/redoc
   - 健康检查: http://127.0.0.1:8000/api/v1/health

### API 速览
- `POST /api/v1/voice-conversion`：声音转换，返回 `audio/wav`
- `POST /api/v1/mix-accompaniment`：伴奏与人声合成，返回 `audio/wav`
- `GET /api/v1/health`：服务健康检查

`/api/v1/voice-conversion` 请求参数：
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `source_audio` | 文件 | 必填，待转换音频 |
| `reference_audio` | 文件 | 必填，参考音色音频 |
| `diffusion_steps` | int | 默认为 10，范围 1~100 |
| `length_adjust` | float | 默认为 1.0，范围 0.5~2.0 |
| `inference_cfg_rate` | float | 默认为 0.7，范围 0~5.0 |
| `auto_f0_adjust` | bool | 默认为 true，自动对齐音高 |
| `pitch_shift` | int | 默认为 0，范围 -12~12，手动升降调 |

`/api/v1/mix-accompaniment` 请求参数：
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `accompaniment_audio` | 文件 | 必填，伴奏音频 |
| `vocal_audio` | 文件 | 必填，人声音频 |
| `accompaniment_gain_db` | float | 默认 0.0，范围 -30~30 dB |
| `vocal_gain_db` | float | 默认 0.0，范围 -30~30 dB |

示例请求（PowerShell）：
```
curl -X POST "http://127.0.0.1:8000/api/v1/voice-conversion" ^
  -F "source_audio=@musics/蓝多多来了.wav" ^
  -F "reference_audio=@references/yae.wav" ^
  -F "diffusion_steps=10" ^
  -F "length_adjust=1.0" ^
  -F "inference_cfg_rate=0.7" ^
  -F "auto_f0_adjust=true" ^
  -F "pitch_shift=0" ^
  -o outputs/result.wav
```

### 客户端脚本
```
.\.venv\Scripts\python.exe client.py `
  --url http://127.0.0.1:8000/api/v1/voice-conversion `
  --source musics/蓝多多来了.wav `
  --reference references/yae.wav `
  --output outputs/client_output.wav `
  --diffusion-steps 10 `
  --length-adjust 1.0 `
  --cfg 0.7 `
  --pitch-shift 0
```

### 模型与缓存
- 首次运行会自动从 Hugging Face 下载 Seed-VC、CampPlus、RMVPE、BigVGAN 等权重并缓存在 `checkpoints/hf_cache`。
- 需要离线运行时，可提前下载完毕后保留该目录，代码已设置 `TRANSFORMERS_OFFLINE=1` 以优先走本地缓存。
- 输出 wav 默认写入 `outputs/`；日志写入 `log/`（loguru + uvicorn）。

## 打包为 Windows 可执行文件
1. 安装 PyInstaller（使用虚拟环境执行）：
```
.\.venv\Scripts\python.exe -m pip install pyinstaller
```
2. 生成单文件可执行程序（包含 uvicorn 日志配置）：
```
.\.venv\Scripts\python.exe -m PyInstaller `
  -F `
  -n SeedVCService `
  --paths api `
  --add-data "uvicorn_config.json;." `
  api\api.py
```
3. 产物位于 `dist/SeedVCService.exe`，使用方式与源码一致，例如：
```
.\dist\SeedVCService.exe --host 0.0.0.0 --port 8000
```
   请保持工作目录包含 `uvicorn_config.json`，并保证 `checkpoints/`、`outputs/` 等目录具有写权限。

## 许可证
本项目采用 MIT License，详见 `LICENSE`。
