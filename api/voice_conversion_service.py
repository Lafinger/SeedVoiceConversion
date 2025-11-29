import threading
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy.io.wavfile import write as wav_write

from seed_conversion import initialize_models, voice_conversion


class VoiceConversionService:
    """负责加载并调度 Seed-VC 声音转换模型的服务类。"""

    _instance: Optional["VoiceConversionService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            logger.debug("VoiceConversionService 已经初始化，返回单例实例")
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        logger.info("开始初始化声音转换模型，请耐心等待……")
        self.device = initialize_models()
        self._lock = threading.Lock()
        self._initialized = True
        logger.info(f"声音转换模型加载完成，当前推理设备: {self.device}")

    def convert(
        self,
        *,
        source_path: Path,
        reference_path: Path,
        diffusion_steps: int = 10,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        output_dir: Path = Path("../outputs"),
        output_filename: Optional[str] = None,
    ) -> Path:
        """
        执行一次声音转换，将 source_path 的内容迁移到 reference_path 的音色。

        参数:
            source_path: 待转换的音频文件路径。
            reference_path: 参考音色文件路径。
            diffusion_steps: 扩散采样步数。
            length_adjust: 输出长度相对源音频的调整系数。
            inference_cfg_rate: classifier-free guidance 的强度。
            auto_f0_adjust: 是否自动对齐音高中位数。
            pitch_shift: 额外的手动升降调（半音单位）。
            output_dir: 输出文件存放目录。
            output_filename: 自定义输出文件名，不包含目录。

        返回:
            输出 wav 文件的完整路径。
        """
        if not source_path.exists():
            raise FileNotFoundError(f"未找到源音频文件: {source_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"未找到参考音频文件: {reference_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_filename:
            output_filename = f"voice_conversion_{uuid.uuid4().hex}.wav"
        output_path = output_dir / output_filename

        logger.info(
            "开始执行声音转换，source=%s, reference=%s, diffusion_steps=%d, length_adjust=%.2f, cfg=%.2f, auto_f0=%s, pitch_shift=%d",
            source_path,
            reference_path,
            diffusion_steps,
            length_adjust,
            inference_cfg_rate,
            auto_f0_adjust,
            pitch_shift,
        )

        with self._lock:
            sample_rate, waveform = voice_conversion(
                source=str(source_path),
                target=str(reference_path),
                diffusion_steps=diffusion_steps,
                length_adjust=length_adjust,
                inference_cfg_rate=inference_cfg_rate,
                auto_f0_adjust=auto_f0_adjust,
                pitch_shift=pitch_shift,
            )

        waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        wav_write(str(output_path), sample_rate, waveform_int16)
        logger.info("声音转换完成，结果保存为 %s", output_path)
        return output_path
