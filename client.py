import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小化 Seed-VC HTTP 客户端示例")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/api/v1/voice-conversion",
        help="声音转换服务的完整地址",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("musics/蓝多多来了.wav"),
        help="待转换的源音频路径",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("references/yae.wav"),
        help="参考音色音频路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("client_output.wav"),
        help="保存结果的输出路径",
    )
    parser.add_argument("--diffusion-steps", type=int, default=50, help="扩散步数")
    parser.add_argument("--length-adjust", type=float, default=1.0, help="输出长度调节系数")
    parser.add_argument("--cfg", type=float, default=0.7, help="CFG 系数")
    parser.add_argument("--pitch-shift", type=int, default=0, help="手动升降调（半音）")
    parser.add_argument(
        "--disable-auto-f0",
        action="store_true",
        help="关闭自动音高对齐（默认开启）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"未找到源音频文件: {args.source}")
    if not args.reference.exists():
        raise FileNotFoundError(f"未找到参考音色文件: {args.reference}")

    files = {
        "source_audio": args.source.open("rb"),
        "reference_audio": args.reference.open("rb"),
    }
    data = {
        "diffusion_steps": args.diffusion_steps,
        "length_adjust": args.length_adjust,
        "inference_cfg_rate": args.cfg,
        "auto_f0_adjust": "false" if args.disable_auto_f0 else "true",
        "pitch_shift": args.pitch_shift,
    }

    try:
        print(f"POST {args.url} ...")
        response = requests.post(args.url, data=data, files=files, timeout=600)
        response.raise_for_status()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(response.content)
        print(f"声音转换成功，结果已保存到: {args.output.resolve()}")
    finally:
        for fh in files.values():
            fh.close()


if __name__ == "__main__":
    main()
