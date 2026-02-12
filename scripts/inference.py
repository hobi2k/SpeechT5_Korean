"""
로컬 아티팩트 기반 SpeechT5 한국어 추론 스크립트.

- model/tokenizer/vocoder/speaker_embedding을 로컬 경로에서 로드한다.
- 텍스트 유틸(korean_text_utils.py)을 로드해 normalize/prosody를 적용한다.

uv run scripts/inference.py \
  --model_dir /mnt/d/tts_data/yae_ko/sp5model \
  --checkpoint_epoch 90 \
  --text "안녕하세요. 학습된 모델 추론 테스트입니다." \
  --out /mnt/d/tts_data/yae_ko/sp5model/test.wav
"""

from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, PreTrainedTokenizerFast


# 추론 디바이스와 기본 샘플레이트.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000


def load_text_utils(model_dir: Path, text_utils_path: Path | None):
    """
    텍스트 전처리 유틸 모듈을 동적으로 로드한다.

    우선순위:
    1. CLI로 지정한 `text_utils_path`
    2. `model_dir/korean_text_utils.py`
    3. 프로젝트 fallback `sp5_kor/text/korean_text_utils.py`

    Args:
        model_dir: 모델 아티팩트 디렉터리.
        text_utils_path: 사용자 지정 유틸 파일 경로. None이면 기본 탐색 순서를 따른다.

    Returns:
        module: normalize_korean/prosody_split/prosody_pause/decompose 함수를 가진 모듈 객체.

    Raises:
        FileNotFoundError: 유틸 파일을 찾지 못한 경우.
        RuntimeError: 모듈 spec 로딩 실패.
    """
    utils_path = text_utils_path if text_utils_path is not None else (model_dir / "korean_text_utils.py")
    if not utils_path.exists():
        utils_path = Path(__file__).resolve().parent.parent / "sp5_kor" / "text" / "korean_text_utils.py"
    if not utils_path.exists():
        raise FileNotFoundError("korean_text_utils.py not found in model_dir or fallback path.")

    spec = importlib.util.spec_from_file_location("korean_text_utils", str(utils_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {utils_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["korean_text_utils"] = module
    spec.loader.exec_module(module)
    return module


def load_assets(
    model_dir: Path,
    model_weights_dir: Path,
    vocoder_dir: Path | None,
    speaker_embedding_path: Path | None,
):
    """
    로컬 모델 추론에 필요한 에셋을 로드한다.

    Args:
        model_dir: tokenizer/vocoder/speaker embedding이 저장된 기본 디렉터리.
        model_weights_dir: SpeechT5 가중치를 로드할 디렉터리.
        vocoder_dir: vocoder 디렉터리 경로. None이면 `model_dir/vocoder`.
        speaker_embedding_path: speaker embedding 파일 경로. None이면 `model_dir/speaker_embedding.pth`.

    Returns:
        tuple: `(model, tokenizer, vocoder, spk_emb)`.

    Raises:
        FileNotFoundError: speaker embedding 파일을 찾지 못한 경우.
    """
    print("Loading local model assets...")

    # 모델은 선택된 체크포인트(또는 기본 model_dir)에서 로드한다.
    model = SpeechT5ForTextToSpeech.from_pretrained(model_weights_dir).to(DEVICE).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

    # vocoder 경로 해석.
    resolved_vocoder_dir = vocoder_dir if vocoder_dir is not None else (model_dir / "vocoder")
    if resolved_vocoder_dir.exists():
        vocoder = SpeechT5HifiGan.from_pretrained(resolved_vocoder_dir).to(DEVICE).eval()
    else:
        # 로컬 vocoder가 없으면 base vocoder를 fallback으로 사용.
        print("[WARN] local vocoder not found. Fallback to microsoft/speecht5_hifigan")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE).eval()

    # speaker embedding 경로 해석.
    spk_path = (
        speaker_embedding_path
        if speaker_embedding_path is not None
        else (model_dir / "speaker_embedding.pth")
    )
    if not spk_path.exists():
        raise FileNotFoundError(f"speaker embedding not found: {spk_path}")

    spk_emb = torch.load(spk_path, map_location=DEVICE)
    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0)
    spk_emb = spk_emb.to(DEVICE)

    print("Assets loaded.")
    return model, tokenizer, vocoder, spk_emb


def resolve_model_weights_dir(model_dir: Path, checkpoint_epoch: int | None) -> Path:
    """
    모델 가중치를 로드할 디렉터리를 결정한다.

    Args:
        model_dir: 기본 모델 디렉터리.
        checkpoint_epoch: 주기 저장 모델 epoch. None이면 기본 모델 디렉터리를 사용한다.

    Returns:
        Path: 실제 모델 가중치 로드 디렉터리.

    Raises:
        FileNotFoundError: 지정한 epoch 모델 디렉터리가 없을 때.
    """
    if checkpoint_epoch is None:
        return model_dir

    ckpt_dir = model_dir / "checkpoints" / f"epoch_{checkpoint_epoch:06d}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint model dir not found: {ckpt_dir}")
    return ckpt_dir


def tts(text: str, out_path: Path, model, tokenizer, vocoder, spk_emb, text_utils) -> None:
    """
    입력 텍스트를 로컬 모델로 합성해 wav 파일로 저장한다.

    Args:
        text: 합성할 원문 텍스트.
        out_path: 출력 wav 파일 경로.
        model: SpeechT5ForTextToSpeech.
        tokenizer: PreTrainedTokenizerFast.
        vocoder: SpeechT5HifiGan.
        spk_emb: speaker embedding 텐서([1, emb_dim]).
        text_utils: korean_text_utils 모듈 객체.

    Returns:
        None
    """
    normalize_korean = text_utils.normalize_korean
    prosody_split = text_utils.prosody_split
    prosody_pause = text_utils.prosody_pause
    decompose_jamo_with_placeholders = text_utils.decompose_jamo_with_placeholders

    print("Normalizing text...")
    norm = normalize_korean(text)

    print("Prosody split...")
    segments = prosody_split(norm)

    # 세그먼트별 파형과 pause를 이어붙인다.
    audio_chunks = []
    for seg in segments:
        print(f"Process segment: {seg}")
        jamo_seq = decompose_jamo_with_placeholders(seg)

        enc = tokenizer(
            jamo_seq,
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            wav = model.generate_speech(
                enc["input_ids"],
                speaker_embeddings=spk_emb,
                vocoder=vocoder,
            )
        audio_chunks.append(wav.cpu().numpy())

        pause = prosody_pause(seg)
        if pause > 0:
            audio_chunks.append(np.zeros(int(SR * pause), dtype=np.float32))

    full_audio = np.concatenate(audio_chunks, axis=0)
    sf.write(str(out_path), full_audio, SR)
    print(f"[Saved] {out_path}")


def parse_args() -> argparse.Namespace:
    """
    CLI 인자를 파싱한다.

    Args:
        없음.

    Returns:
        argparse.Namespace: 추론 실행 인자 객체.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./speecht5_kss_korean_jamo"),
        help="로컬 모델 폴더 경로",
    )
    parser.add_argument(
        "--text_utils_path",
        type=Path,
        default=None,
        help="korean_text_utils.py 경로 (기본: model_dir/korean_text_utils.py)",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=Path,
        default=None,
        help="vocoder 디렉터리 경로 (기본: model_dir/vocoder)",
    )
    parser.add_argument(
        "--speaker_embedding_path",
        type=Path,
        default=None,
        help="speaker_embedding.pth 경로 (기본: model_dir/speaker_embedding.pth)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="안녕하세요. 오늘은 3.5km를 걸었습니다! 숫자와 단위 처리 테스트입니다.",
        help="합성할 텍스트",
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="주기 저장 모델 epoch 선택 (예: 40 -> model_dir/checkpoints/epoch_000040)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_local.wav"),
        help="출력 wav 경로",
    )
    return parser.parse_args()


def main() -> None:
    """
    로컬 추론 파이프라인을 실행한다.

    Args:
        없음.

    Returns:
        None
    """
    args = parse_args()
    model_weights_dir = resolve_model_weights_dir(args.model_dir, args.checkpoint_epoch)
    print(f"Selected model weights dir: {model_weights_dir}")

    text_utils = load_text_utils(args.model_dir, args.text_utils_path)
    model, tokenizer, vocoder, spk_emb = load_assets(
        args.model_dir,
        model_weights_dir,
        args.vocoder_dir,
        args.speaker_embedding_path,
    )
    tts(args.text, args.out, model, tokenizer, vocoder, spk_emb, text_utils)


if __name__ == "__main__":
    main()
