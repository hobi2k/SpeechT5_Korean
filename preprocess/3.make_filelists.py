from pathlib import Path
import json
import re

import pandas as pd

# 원본 데이터 루트.
DATA_ROOT = Path("/mnt/d/tts_data/yae_ko/")
# 리샘플링된 wav 파일 디렉터리.
WAV_DIR = DATA_ROOT / "sp5"
# 원본 메타데이터 CSV.
CSV_PATH = DATA_ROOT / "metadata_raw.csv"
# 생성 파일 출력 디렉터리.
OUT_DIR = DATA_ROOT / "sp5file"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# README에 정의한 JSONL 스키마:
# {"audio_path":"...","speaker":"...","language":"...","text":"..."}
OUT_JSONL = OUT_DIR / "metadata.jsonl"

# 고정 메타 필드(단일 화자/단일 언어 세트).
SPEAKER = "yae"
LANGUAGE = "ko"


def normalize_text(text: str) -> str:
    """
    CSV 원문 텍스트를 JSONL 적재 전 형태로 정규화한다.

    처리 규칙:
    1. 줄바꿈/캐리지리턴 제거
    2. 연속 공백 압축

    Args:
        text: 원본 텍스트 값.

    Returns:
        str: 공백 정리가 끝난 텍스트.
    """
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def has_korean(text: str) -> bool:
    """
    문자열에 한글 음절 블록(가-힣)이 포함되는지 검사한다.

    Args:
        text: 검사 대상 문자열.

    Returns:
        bool: 한글 포함 여부.
    """
    return bool(re.search(r"[가-힣]", text))


# CSV 로드.
df = pd.read_csv(CSV_PATH)

# 최종 JSONL 레코드 버퍼.
records: list[dict] = []
for _, row in df.iterrows():
    # CSV에 저장된 wav 경로에서 파일명만 추출해 표준 wav 디렉터리와 결합.
    wav_name = Path(row["wav"]).name
    wav_path = WAV_DIR / wav_name
    # 실제 파일이 없는 샘플은 제외.
    if not wav_path.exists():
        continue

    # 텍스트 정규화 후 빈 문장이면 제외.
    text = normalize_text(row["text"])
    if not text:
        continue
    # 한글이 전혀 없는 문장은 본 파이프라인에서 제외.
    if not has_korean(text):
        continue

    # 학습용 JSONL 스키마로 레코드 구성.
    records.append(
        {
            "audio_path": str(wav_path),
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "text": text,
        }
    )

# UTF-8 JSONL로 저장(한글 유지 위해 ensure_ascii=False).
with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"jsonl: {len(records)} -> {OUT_JSONL}")
