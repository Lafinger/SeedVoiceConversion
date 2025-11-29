import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from scipy.io.wavfile import write as wav_write

from hf_utils import load_custom_model_from_hf
from modules.audio import mel_spectrogram
from modules.campplus.DTDNN import CAMPPlus
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.hifigan.f0_predictor import ConvRNNF0Predictor
from modules.hifigan.generator import HiFTGenerator
from modules.rmvpe import RMVPE

# 默认半精度推理，老显卡修改为: torch.float32
dtype = torch.float16

class ConversionCancelled(Exception):
    """转换被取消时抛出的异常。"""

# 使用本地缓存模型
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

# 以下为全局运行态缓存，初始化一次后供整个服务多次调用，避免重复加载
device: torch.device = torch.device("cpu")
model_f0 = None
semantic_fn = None
vocoder_fn = None
campplus_model = None
to_mel_f0 = None
mel_fn_args = None
f0_fn = None
overlap_wave_len = None
max_context_window = None
sr = None
hop_length = None
overlap_frame_len = 16


def load_models(args) -> Tuple:
    """
    加载推理所需的核心模型与函数，包括DiT主体、语义编码器、声码器以及F0分析器。

    参数:
        args: 包含checkpoint以及配置路径的命名空间对象，允许用户覆盖默认模型。

    返回:
        tuple: 由多个推理组件和函数构成的元组，供初始化流程直接解包使用。
    """
    global sr, hop_length
    print(f"使用设备: {device}")

    # 若用户未指定checkpoint，默认从HF下载官方模型；否则使用用户提供的路径
    if not args.checkpoint:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )
    else:
        print(f"使用自定义 checkpoint: {args.checkpoint}")
        dit_checkpoint_path = args.checkpoint
        dit_config_path = args.config
    # 读取模型配置以获取声学参数、声码器信息等
    config = yaml.safe_load(open(dit_config_path, "r", encoding="utf-8"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # 将权重加载至模型并切换到推理模式
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    # 多分支模块需逐个放入device并启用eval模式
    for key in model:
        model[key].eval()
        model[key].to(device)
    # 预先为CFM估计器创建缓存，避免每次推理重复分配
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # 加载语者表示模型CAMPPlus，用于抽取参考音色嵌入
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus.eval()
    campplus.to(device)

    # 声码器负责将Mel或特征映射回波形，不同类型初始化方式不同
    vocoder_type = model_params.vocoder.type
    if vocoder_type == "bigvgan":
        from modules.bigvgan import bigvgan

        # BigVGAN 直接载入官方提供的预训练权重
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder = bigvgan_model
    elif vocoder_type == "hifigan":
        hift_config = yaml.safe_load(open("configs/hifigan.yml", "r", encoding="utf-8"))
        # HiFi-GAN 变体依赖额外的f0预测器，需读取独立配置
        hift_gen = HiFTGenerator(
            **hift_config["hift"],
            f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
        )
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r", encoding="utf-8"))
        vocos_path = model_params.vocoder.vocos.path
        # Vocos 属于模块化结构，需要根据配置构建模型再加载权重
        vocos_model_params = recursive_munch(vocos_config["model_params"])
        vocos = build_model(vocos_model_params, stage="mel_vocos")
        vocos, _, _, _ = load_checkpoint(
            vocos,
            None,
            vocos_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        vocoder = vocos.decoder
    else:
        raise ValueError(f"未知声码器类型: {vocoder_type}")

    # 语义分词器（speech tokenizer）决定如何从音频中抽取高层语义表征
    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        # Whisper 仅需编码器部分，decoder可以释放内存
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=dtype).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            # 统一提取16k音频的对数Mel特征，再送入Whisper编码器获取语义隐藏状态
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == "cnhubert":
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        hubert_model_name = model_params.speech_tokenizer.name
        # 华语数据常用CN-HuBERT，需配套的特征提取器
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            # 将batch内音频逐条转换为numpy，再统一padding输入模型
            ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
            ori_inputs = hubert_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == "xlsr":
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_name = model_params.speech_tokenizer.name
        output_layer = model_params.speech_tokenizer.output_layer
        # XLSR 使用多语言wav2vec2模型，可根据配置裁剪编码层
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            # XLSR同样需要对输入进行padding，保证跨batch长度一致
            ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"未知语音分词器类型: {speech_tokenizer_type}")

    # 记录梅尔谱计算所需的参数，确保推理与训练阶段设置一致
    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000,
        "center": False,
    }

    # 懒加载一个mel_spectrogram函数，供外部以相同参数快速调用
    mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    # RMVPE负责从波形中估算F0曲线，音高处理的质量取决于此
    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    f0_fn = rmvpe.infer_from_audio

    return model, semantic_fn, vocoder, campplus, mel_fn, mel_fn_args, f0_fn


def adjust_f0_semitones(f0_sequence, n_semitones):
    """
    依据半音数量平移F0序列（对数刻度下每12半音等于频率翻倍）。

    参数:
        f0_sequence: 原始F0序列（tensor）。
        n_semitones: 需要升降的半音数，可正可负。

    返回:
        调整后的F0序列。
    """
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    """
    对相邻音频片段做交叉淡入淡出，避免拼接边界产生爆音。

    参数:
        chunk1: 先前片段的波形数组。
        chunk2: 当前片段的波形数组。
        overlap: 交叉的采样点数。

    返回:
        已经平滑过渡的chunk2数组。
    """
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(
    source: str,
    target: str,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
    auto_f0_adjust: bool,
    pitch_shift: int,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[int, np.ndarray]:
    """
    执行一次端到端的声音转换流程，将源语音的内容迁移到参考音色上。

    参数:
        source: 待转换语音路径。
        target: 参考音色语音路径。
        diffusion_steps: 扩散采样步数，越大越稳。
        length_adjust: 控制输出长度与输入的比例。
        inference_cfg_rate: classifier-free guidance的力度系数，平衡保真与风格。
        auto_f0_adjust: 是否根据参考音调整目标音高中位数。
        pitch_shift: 额外的手动升降调（半音）。

    返回:
        采样率与最终合成的波形数组。
    """
    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionCancelled("conversion cancelled")

    _check_cancel()
    inference_module = model_f0
    # 读取源音频与参考音频，并统一采样率
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # 转换为tensor并限制参考音频最大时长，避免过长占用显存
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # 模型的语义分词器基于16k音频，因此需要重采样
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    # 对长音频需要切块提取语义特征，避免超出模型上下文
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        # 采用滑窗+重叠的方式拼接语义特征，保证跨块连续
        overlapping_time = 5
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1,
                )
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    # 参考音色同样需要语义特征，供prompt条件使用
    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    # 计算源音频与参考音频的梅尔谱，后续输入扩散模型和声码器
    mel = to_mel_f0(source_audio.to(device).float())
    mel2 = to_mel_f0(ref_audio.to(device).float())

    # 长度控制：源语音允许按比例放缩，参考音需保持原始长度
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # 使用Kaldi的fbank特征提取器为CAMPPlus生成输入
    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    # 统计归一化能提升风格提取稳定性
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    # CAMPPlus输出的嵌入描述参考说话人的音色
    style2 = campplus_model(feat2.unsqueeze(0))

    # RMVPE返回numpy数组，后续会被送入扩散模型的长度调节器
    F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    # 不同设备类型对半精度支持不同，这里做兼容转换
    if device.type == "mps":
        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
    else:
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    # 仅保留有声帧的F0用于统计，否则噪声会影响中位数估计
    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    # 在对数域上计算中位数，便于进行加性平移
    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    shifted_log_f0_alt = log_f0_alt.clone()
    # 如果开启自动音高对齐，则将源音频的中位数迁移到参考音高
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

    # 通过长度调节器将语义特征对齐到目标梅尔长度，并附加调制后的F0
    cond, _, _, _, _ = inference_module.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    # prompt_condition 使用参考音色，作为前缀以注入音色信息
    prompt_condition, _, _, _, _ = inference_module.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )
    # 将F0序列插值到与条件长度一致，确保逐帧对应
    interpolated_shifted_f0_alt = torch.nn.functional.interpolate(
        shifted_f0_alt.unsqueeze(1),
        size=cond.size(1),
        mode="nearest",
    ).squeeze(1)
    # 每次拼接时为prompt预留上下文窗口，剩余部分用于逐块生成
    max_source_window = max_context_window - mel2.size(2)

    # 记录当前已处理的帧数，并缓存生成的音频片段和上一个尾部
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None
    _check_cancel()

    # 采用滑动窗口形式迭代生成目标mel，再交由声码器合成波形
    while processed_frames < cond.size(1):
        _check_cancel()
        # 当前窗口的语义条件，并判断是否已到末尾
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        # 与参考prompt拼接后送入扩散模型
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        # autocast结合半精度可以显著降低显存占用
        with torch.autocast(device_type=device.type, dtype=dtype):
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        # 将预测的mel片段送入声码器得到波形
        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()
        if vc_wave.ndim == 1:
            # 声码器可能返回单通道张量，需要保证存在batch维
            vc_wave = vc_wave.unsqueeze(0)
        if processed_frames == 0:
            # 第一块没有历史尾巴，直接截断或整块输出
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            # 最后一块与前一块交叉淡入淡出后直接结束
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            # 中间块需要裁切尾部并与上一块连接
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    # 拼接所有片段得到最终波形
    final_wave = np.concatenate(generated_wave_chunks)
    return sr, final_wave


def initialize_models(preferred_device: Optional[torch.device] = None) -> torch.device:
    """
    根据硬件情况挑选可用设备，并初始化所有模型组件。

    参数:
        preferred_device: 用户强制指定的设备；若为空则自动探测。

    返回:
        torch.device: 最终用于推理的设备对象。
    """
    global device, model_f0, semantic_fn, vocoder_fn, campplus_model, to_mel_f0, mel_fn_args, f0_fn
    global overlap_wave_len, max_context_window
    if preferred_device is None:
        # 优先使用CUDA，其次是Apple MPS，最后退回CPU
        if torch.cuda.is_available():
            preferred_device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            preferred_device = torch.device("mps")
        else:
            preferred_device = torch.device("cpu")
    device = preferred_device

    args = SimpleNamespace(checkpoint=None, config=None)
    # 加载所有推理组件，并缓存至全局变量
    (
        model_f0,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel_f0,
        mel_fn_args,
        f0_fn,
    ) = load_models(args)

    # 根据采样率和hop_length计算上下文窗口大小及交叉淡入长度
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    return device


def run_single_conversion(source_audio: Path, reference_audio: Path, output_wav: Path) -> None:
    """
    提供一个极简调用示例：从音频路径读取、执行转换，并将结果写入文件。

    参数:
        source_audio: 待转换音频路径。
        reference_audio: 参考音色音频路径。
        output_wav: 输出wav文件路径。
    """
    # 调用主转换函数，使用一组默认参数
    sample_rate, waveform = voice_conversion(
        source=str(source_audio),
        target=str(reference_audio),
        diffusion_steps=10,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        auto_f0_adjust=True,
        pitch_shift=0,
    )
    # 将浮点波形裁剪并量化为16-bit PCM，再写入磁盘
    waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wav_write(str(output_wav), sample_rate, waveform_int16)


if __name__ == "__main__":
    SOURCE_AUDIO = Path("../musics/蓝多多来了.wav")
    REFERENCE_AUDIO = Path("../references/yae.wav")
    OUTPUT_AUDIO = Path("../outputs/output.wav")

    # 初始化并缓存所有模型，后续直接复用
    device = initialize_models()
    print(f"使用设备: {device}")
    run_single_conversion(SOURCE_AUDIO, REFERENCE_AUDIO, OUTPUT_AUDIO)
    print(f"转换完成，结果保存在 {OUTPUT_AUDIO}")
