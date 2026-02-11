"""
JSONL 기반 SpeechT5 학습 엔트리 스크립트.

예시:
uv run python -m scripts.train \
  --jsonl_path /mnt/d/tts_data/yae_ko/sp5file/metadata.jsonl \
  --output_dir /mnt/d/tts_data/yae_ko/sp5model \
  --min_audio_len 0.8 \
  --max_audio_len 10 \
  --batch_size 2 \
  --min_tokens 5 \
  --max_tokens 200 \
  --min_token_per_sec 2.0 \
  --max_token_per_sec 35.0
  --resume
"""

import argparse
from pathlib import Path

import torch

from sp5_kor import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    """
    CLI 인자를 파싱한다.

    Args:
        없음.

    Returns:
        argparse.Namespace: 학습 실행에 필요한 모든 CLI 옵션이 담긴 객체.
    """
    p = argparse.ArgumentParser(
        description="Train SpeechT5 Korean model from local JSONL dataset."
    )

    # 필수 입력 경로.
    p.add_argument("--jsonl_path", type=Path, required=True, help="JSONL path")
    p.add_argument("--output_dir", type=Path, required=True, help="Artifact output dir")

    # 선택 입력 경로.
    p.add_argument(
        "--jamo_vocab_path",
        type=Path,
        default=Path("./sp5_kor/text/jamo_vocab.txt"),
        help="Path to jamo vocab txt",
    )

    # 학습 하이퍼파라미터.
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)

    # 정렬 안정성 필터 파라미터.
    p.add_argument("--min_audio_len", type=float, default=0.8)
    p.add_argument("--max_audio_len", type=int, default=10)
    p.add_argument("--min_tokens", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--min_token_per_sec", type=float, default=2.0)
    p.add_argument("--max_token_per_sec", type=float, default=35.0)

    # 기타 설정.
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--train_ratio", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", action="store_true", help="Resume from output_dir/checkpoint_last.pt")
    return p.parse_args()


def main() -> None:
    """
    학습 설정을 구성하고 학습 파이프라인을 실행한다.

    Args:
        없음.

    Returns:
        None
    """
    args = parse_args()

    # device='auto'면 CUDA 가용 여부로 결정.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # dataclass 설정 객체로 변환.
    cfg = TrainingConfig(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        jamo_vocab_path=args.jamo_vocab_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_audio_len=args.min_audio_len,
        max_audio_len=args.max_audio_len,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_token_per_sec=args.min_token_per_sec,
        max_token_per_sec=args.max_token_per_sec,
        num_proc=args.num_proc,
        train_ratio=args.train_ratio,
        seed=args.seed,
        device=device,
        resume=args.resume,
    )

    # 학습 실행.
    run_training(cfg)


if __name__ == "__main__":
    main()
