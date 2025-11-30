import uuid
from pathlib import Path
from typing import Optional

from loguru import logger
from pydub import AudioSegment


def _prepare_segment(
    segment: AudioSegment,
    *,
    frame_rate: int,
    channels: int,
    sample_width: int,
) -> AudioSegment:
    """
    将音频在采样率、声道数和位深上对齐，避免 overlay 时格式不一致导致的错误。
    """
    return (
        segment.set_frame_rate(frame_rate)
        .set_channels(channels)
        .set_sample_width(sample_width)
    )


def merge_accompaniment_and_vocals(
    *,
    accompaniment_path: Path,
    vocal_path: Path,
    output_dir: Path = Path("../outputs"),
    output_filename: Optional[str] = None,
    accompaniment_gain_db: float = 0.0,
    vocal_gain_db: float = 0.0,
) -> Path:
    """
    合并伴奏和人声轨道并导出 wav 文件。

    参数:
        accompaniment_path: 伴奏音频文件路径。
        vocal_path: 人声音频文件路径。
        output_dir: 输出目录，默认写入项目根目录下的 outputs。
        output_filename: 输出文件名，不包含目录，未指定则自动生成。
        accompaniment_gain_db: 伴奏增益（dB），用于调节伴奏响度。
        vocal_gain_db: 人声增益（dB），用于调节人声响度。

    返回:
        合并后 wav 文件的完整路径。
    """
    if not accompaniment_path.exists():
        raise FileNotFoundError(f"未找到伴奏文件: {accompaniment_path}")
    if not vocal_path.exists():
        raise FileNotFoundError(f"未找到人声文件: {vocal_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_filename:
        output_filename = f"merged_audio_{uuid.uuid4().hex}.wav"
    output_path = output_dir / output_filename

    logger.info(
        "开始合并伴奏与人声，伴奏=%s，人声=%s，伴奏增益=%.1f，人声增益=%.1f",
        accompaniment_path,
        vocal_path,
        accompaniment_gain_db,
        vocal_gain_db,
    )

    accompaniment = AudioSegment.from_file(accompaniment_path)
    vocals = AudioSegment.from_file(vocal_path)

    target_frame_rate = accompaniment.frame_rate or vocals.frame_rate or 44100
    target_channels = accompaniment.channels or vocals.channels or 2
    target_sample_width = accompaniment.sample_width or vocals.sample_width or 2

    accompaniment = _prepare_segment(
        accompaniment,
        frame_rate=target_frame_rate,
        channels=target_channels,
        sample_width=target_sample_width,
    )
    vocals = _prepare_segment(
        vocals,
        frame_rate=target_frame_rate,
        channels=target_channels,
        sample_width=target_sample_width,
    )

    if accompaniment_gain_db:
        accompaniment = accompaniment.apply_gain(accompaniment_gain_db)
    if vocal_gain_db:
        vocals = vocals.apply_gain(vocal_gain_db)

    merged = accompaniment.overlay(vocals)
    merged.export(output_path, format="wav")

    logger.info("伴奏与人声合并完成，输出: %s", output_path)
    return output_path
