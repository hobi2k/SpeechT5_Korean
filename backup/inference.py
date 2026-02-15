"""
SpeechT5 Korean TTS Inference Script (Final)

Hugging Face Hub에 업로드된 한국어 SpeechT5 TTS 모델을 다운로드하여
텍스트 -> 음성(wav) 변환을 수행한다.

핵심 특징:
1. 자모(Jamo) 기반 WordLevel 토크나이저 사용
2. 훈련과 동일한 korean_text_utils.py를 Hub에서 직접 로드
3. 추론 단계에서만 숫자/단위 발음 확정(normalize)
4. prosody 기반 문장 분할 + pause 삽입
5. SpeechT5 + HiFi-GAN vocoder
6. 단일 화자 speaker embedding 로드

즉, 이 코드는:
“학습 파이프라인의 철학을 그대로 유지한 채
실제 사람이 듣기 좋은 음성을 생성하는 코드”
이다.
"""

# 기본 라이브러리

import numpy as np
import torch
import soundfile as sf # wav 파일 저장

# Python 표준 방식의 "동적 import"
import importlib.util
import sys

from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    PreTrainedTokenizerFast,
)

from huggingface_hub import hf_hub_download

# Hugging Face Hub에 업로드한 모델 ID
# (model, tokenizer, speaker_embedding, korean_text_utils.py가 모두 여기에 있음)
MODEL_ID = "ahnhs2k/speecht5-korean-jamo"

# GPU 사용 가능하면 CUDA, 아니면 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 샘플레이트
SR = 16000


# korean_text_utils.py를 Hub에서 직접 로드
"""
- 훈련 시 사용한 텍스트 전처리 코드와
  추론 시 사용하는 코드가 조금이라도 다르면
  발음, 억양, 정렬이 깨질 수 있다.

- 따라서 모델과 함께 업로드된 korean_text_utils.py를
  Hub에서 직접 다운로드하여 런타임 import 한다.

장점:
- 훈련/추론 전처리 100% 동기화
- 버전 불일치 방지
- 재현성 보장
"""

# Hub에서 korean_text_utils.py 다운로드
utils_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="korean_text_utils.py",
)

# 파일 경로로부터 모듈 spec 생성
spec = importlib.util.spec_from_file_location(
    "korean_text_utils",
    utils_path
)

# 모듈 객체 생성
korean_text_utils = importlib.util.module_from_spec(spec)

# sys.modules에 등록 (다른 모듈에서도 import 가능)
sys.modules["korean_text_utils"] = korean_text_utils

# 실제 코드 실행
spec.loader.exec_module(korean_text_utils)

# 필요한 텍스트 유틸 함수 로드

# 훈련용 placeholder 전처리
inject_tokens_for_training = korean_text_utils.inject_tokens_for_training

# placeholder(<num>, <comma> 등)를 보존하면서 한글을 자모로 분해
decompose_jamo_with_placeholders = korean_text_utils.decompose_jamo_with_placeholders

# 추론용 텍스트 정규화
# 숫자/단위/퍼센트 등을 실제 발음 문자열로 변환
normalize_korean = korean_text_utils.normalize_korean

# 구두점 기반 문장 분할 (prosody segmentation)
prosody_split = korean_text_utils.prosody_split

# 구두점에 따른 pause 길이 결정
prosody_pause = korean_text_utils.prosody_pause


# 모델 / 토크나이저 / Vocoder / Speaker Embedding 로드
def load_assets():
    """
    Hugging Face Hub에서 TTS 추론에 필요한 모든 에셋을 로드한다.

    로드 대상:
    1. SpeechT5ForTextToSpeech (텍스트 -> mel)
    2. PreTrainedTokenizerFast (자모 WordLevel 토크나이저)
    3. HiFi-GAN vocoder (mel -> waveform)
    4. speaker_embedding.pth (단일 화자 음색)

    반환:
    - model
    - tokenizer
    - vocoder
    - speaker_embedding (shape: [1, emb_dim])
    """
    print("Loading model / tokenizer / vocoder / speaker embedding...")

    # TTS 모델 로드
    # eval(): inference 모드
    model = (
        SpeechT5ForTextToSpeech
        .from_pretrained(MODEL_ID)
        .to(DEVICE)
        .eval()
    )

    # 토크나이저 로드
    # 훈련 시 사용한 자모 WordLevel tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)

    # Vocoder 로드
    # mel spectrogram을 실제 파형으로 변환
    vocoder = (
        SpeechT5HifiGan
        .from_pretrained("microsoft/speecht5_hifigan")
        .to(DEVICE)
        .eval()
    )

    # peaker Embedding 로드
    # 단일 화자이므로 항상 동일 embedding 사용
    spk_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="speaker_embedding.pth"
    )
    spk_emb = torch.load(spk_path, map_location=DEVICE)

    # shape 보정:
    # (emb_dim,) → (1, emb_dim)
    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0)

    spk_emb = spk_emb.to(DEVICE)

    print("Assets loaded.")
    return model, tokenizer, vocoder, spk_emb


# 실제 로드 실행
model, tokenizer, vocoder, spk_emb = load_assets()


# Prosody 기반 TTS 함수
def tts(text: str, out_path="output.wav"):
    """
    텍스트를 음성으로 변환하는 메인 함수.

    파이프라인:
    1. normalize_korean (추론용 발음 확정)
    2. prosody_split (문장/구두점 단위 분할)
    3. 각 segment에 대해:
       - 자모 분해
       - 토크나이즈
       - SpeechT5 + HiFi-GAN으로 음성 생성
       - 구두점 기반 pause 삽입
    4. 모든 오디오 조각을 concat
    5. wav 파일 저장
    """

    # 텍스트 정규화 (추론 전용)
    print("Normalizing text...")
    norm = normalize_korean(text)

    # Prosody 기반 문장 분할
    print("Prosody split...")
    segments = prosody_split(norm)

    # 최종 오디오 조각들을 담을 리스트
    audio_chunks = []

    # Segment 단위 TTS
    for seg in segments:
        print(f"Process segment: {seg}")

        # placeholder-aware 자모 분해
        # - <num>, <comma> 등은 그대로 유지
        # - 한글 음절은 초/중/종성 자모로 분해
        jamo_seq = decompose_jamo_with_placeholders(seg)

        # 토크나이저 호출
        # - is_split_into_words=True:
        # - jamo_seq의 각 원소를 "이미 분해된 토큰"으로 간주
        enc = tokenizer(
            jamo_seq,
            is_split_into_words=True,
            add_special_tokens=True,   # <bos>, <eos> 추가
            return_tensors="pt",
        )

        # GPU/CPU 이동
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # SpeechT5 inference
        # - generate_speech 내부에서:
        #   text -> mel -> vocoder -> waveform
        with torch.no_grad():
            wav = model.generate_speech(
                enc["input_ids"],
                speaker_embeddings=spk_emb,
                vocoder=vocoder,
            )

        # numpy로 변환하여 저장
        audio_chunks.append(wav.cpu().numpy())

        # Prosody 기반 pause 삽입
        pause = prosody_pause(seg)
        if pause > 0:
            silence = np.zeros(
                int(SR * pause),
                dtype=np.float32
            )
            audio_chunks.append(silence)

    # 전체 오디오 병합 및 저장
    full_audio = np.concatenate(audio_chunks, axis=0)
    sf.write(out_path, full_audio, SR)

    print(f"[Saved] {out_path}")
    return out_path


# 테스트 실행
if __name__ == "__main__":
    text = "안녕하세요. 오늘은 3.5km를 걸었습니다! 숫자와 단위 처리 테스트입니다."
    tts(text, "demo.wav")
