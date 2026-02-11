"""
SpeechT5 Korean TTS Inference (Local Artifacts)

로컬에 저장된 학습 결과 폴더를 직접 읽어
텍스트 -> 음성(wav) 합성을 수행한다.
"""

from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, PreTrainedTokenizerFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000


def load_text_utils(model_dir: Path):
    """
    모델 폴더의 korean_text_utils.py를 우선 사용한다.
    없으면 현재 작업 디렉터리의 korean_text_utils.py를 fallback으로 사용한다.
    """
    utils_path = model_dir / "korean_text_utils.py"
    if not utils_path.exists():
        utils_path = Path("korean_text_utils.py")
    if not utils_path.exists():
        raise FileNotFoundError(
            "korean_text_utils.py not found in model_dir or current directory."
        )

    spec = importlib.util.spec_from_file_location("korean_text_utils", str(utils_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {utils_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["korean_text_utils"] = module
    spec.loader.exec_module(module)
    return module


def load_assets(model_dir: Path):
    """
    로컬 모델 폴더에서 추론 에셋을 로드한다.
    - model_dir/: SpeechT5 model + tokenizer
    - model_dir/vocoder/: vocoder (있으면 사용)
    - model_dir/speaker_embedding.pth
    """
    print("Loading local model assets...")

    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).to(DEVICE).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)

    vocoder_dir = model_dir / "vocoder"
    if vocoder_dir.exists():
        vocoder = SpeechT5HifiGan.from_pretrained(vocoder_dir).to(DEVICE).eval()
    else:
        print("[WARN] local vocoder not found. Fallback to microsoft/speecht5_hifigan")
        vocoder = (
            SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            .to(DEVICE)
            .eval()
        )

    spk_path = model_dir / "speaker_embedding.pth"
    if not spk_path.exists():
        raise FileNotFoundError(f"speaker embedding not found: {spk_path}")
    spk_emb = torch.load(spk_path, map_location=DEVICE)
    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0)
    spk_emb = spk_emb.to(DEVICE)

    print("Assets loaded.")
    return model, tokenizer, vocoder, spk_emb


def tts(text: str, out_path: Path, model, tokenizer, vocoder, spk_emb, text_utils):
    normalize_korean = text_utils.normalize_korean
    prosody_split = text_utils.prosody_split
    prosody_pause = text_utils.prosody_pause
    decompose_jamo_with_placeholders = text_utils.decompose_jamo_with_placeholders

    print("Normalizing text...")
    norm = normalize_korean(text)

    print("Prosody split...")
    segments = prosody_split(norm)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./speecht5_kss_korean_jamo"),
        help="로컬 모델 폴더 경로",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="안녕하세요. 오늘은 3.5km를 걸었습니다! 숫자와 단위 처리 테스트입니다.",
        help="합성할 텍스트",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_local.wav"),
        help="출력 wav 경로",
    )
    args = parser.parse_args()

    text_utils = load_text_utils(args.model_dir)
    model, tokenizer, vocoder, spk_emb = load_assets(args.model_dir)
    tts(args.text, args.out, model, tokenizer, vocoder, spk_emb, text_utils)


if __name__ == "__main__":
    main()
