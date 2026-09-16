import json
import os
from pathlib import Path

from huggingface_hub import constants, hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger


MODEL_CACHE_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _offline_mode():
    return constants.HF_HUB_OFFLINE or any(
        os.environ.get(name, "").upper() in {"1", "ON", "YES", "TRUE"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _cached_file(repo_id, filename):
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename,
            cache_dir=str(MODEL_CACHE_DIR), local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None


def _require_download_allowed(repo_id, missing):
    message = f"模型 {repo_id} 缺少文件 {', '.join(missing)}；缓存目录: {MODEL_CACHE_DIR}"
    if _offline_mode():
        logger.error("{}；环境变量已启用离线模式，无法下载", message)
        raise LocalEntryNotFoundError(f"{message}；环境变量已启用离线模式，无法下载")
    logger.info("{}；开始自动下载", message)


def _load_file_from_hf(repo_id, filename):
    cached = _cached_file(repo_id, filename)
    if cached is not None:
        logger.info("使用本地模型文件: {} / {}；路径: {}", repo_id, filename, cached)
        return cached

    _require_download_allowed(repo_id, [filename])
    try:
        return hf_hub_download(
            repo_id=repo_id, filename=filename,
            cache_dir=str(MODEL_CACHE_DIR), local_files_only=False,
        )
    except Exception:
        logger.exception("模型下载失败: {} / {}；缓存目录: {}", repo_id, filename, MODEL_CACHE_DIR)
        raise


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    model_path = _load_file_from_hf(repo_id, model_filename)
    if config_filename is None:
        return model_path
    config_path = _load_file_from_hf(repo_id, config_filename)
    return model_path, config_path


def _missing_weights(repo_id):
    # 与 Transformers 的默认选择顺序一致；分片索引存在时还必须检查每个分片。
    candidates = (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )
    for filename in candidates:
        cached = _cached_file(repo_id, filename)
        if cached is None:
            continue
        if not filename.endswith(".index.json"):
            return []
        with open(cached, encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(shard, str) and shard for shard in weight_map.values()
        ):
            raise ValueError(f"模型分片索引无效: {cached}")
        return [shard for shard in sorted(set(weight_map.values())) if _cached_file(repo_id, shard) is None]
    return ["模型权重（model.safetensors / pytorch_model.bin 或对应分片索引）"]


def load_pretrained_from_hf(
    loader, model_name_or_path, *, required_files=("config.json",), check_weights=False, **kwargs
):
    """检查当前组件的必要文件，仅在缓存缺失时允许 from_pretrained 联网。"""
    model_id = str(model_name_or_path)
    local_path = Path(model_id).expanduser()
    is_local = (
        isinstance(model_name_or_path, os.PathLike)
        or local_path.exists()
        or local_path.is_absolute()
        or model_id.startswith(("./", "../", ".\\", "..\\", "~"))
    )
    if is_local:
        if not local_path.exists():
            raise FileNotFoundError(f"未找到本地模型路径: {local_path}")
        logger.info("使用指定的本地模型: {}", local_path)
        return loader(str(local_path), cache_dir=str(MODEL_CACHE_DIR), local_files_only=True, **kwargs)

    missing = [filename for filename in required_files if _cached_file(model_id, filename) is None]
    if check_weights:
        missing.extend(_missing_weights(model_id))
    if missing:
        _require_download_allowed(model_id, missing)
    else:
        logger.info("使用本地模型缓存: {}；目录: {}", model_id, MODEL_CACHE_DIR)

    try:
        return loader(model_id, cache_dir=str(MODEL_CACHE_DIR), local_files_only=not missing, **kwargs)
    except Exception:
        # 加载失败只记录并保留原异常，不能将损坏文件或运行时错误当作缓存缺失重试。
        logger.exception("模型获取或加载失败: {}；缓存目录: {}", model_id, MODEL_CACHE_DIR)
        raise
