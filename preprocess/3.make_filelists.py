from pathlib import Path
import json
import re

import pandas as pd

# paths
DATA_ROOT = Path("/mnt/d/tts_data/yae_ko/")
WAV_DIR = DATA_ROOT / "sp5"
CSV_PATH = DATA_ROOT / "metadata_raw.csv"
OUT_DIR = DATA_ROOT / "sp5file"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# README JSONL format
# {"audio_path":"...","speaker":"...","language":"...","text":"..."}
OUT_JSONL = OUT_DIR / "metadata.jsonl"

SPEAKER = "yae"
LANGUAGE = "ko"


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


df = pd.read_csv(CSV_PATH)

records: list[dict] = []
for _, row in df.iterrows():
    wav_name = Path(row["wav"]).name
    wav_path = WAV_DIR / wav_name
    if not wav_path.exists():
        continue

    text = normalize_text(row["text"])
    if not text:
        continue
    if not has_korean(text):
        continue

    records.append(
        {
            "audio_path": str(wav_path),
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "text": text,
        }
    )

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"jsonl: {len(records)} -> {OUT_JSONL}")

