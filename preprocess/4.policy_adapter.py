from pathlib import Path
from typing import Optional, Tuple
import json
import re

# paths
DATA_ROOT = Path("/mnt/d/tts_data/yae_ko/")
WAV_DIR = DATA_ROOT / "sp5"
FILELIST_DIR = DATA_ROOT / "sp5file"
META_PATH = FILELIST_DIR / "metadata.jsonl"

REQUIRED_KEYS = {"audio_path", "speaker", "language", "text"}


def filter_and_clean_tts_text(text: str) -> Tuple[str, Optional[str]]:
    if re.search(r"\{[^}]+\}", text):
        return "DROP", None

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[「」『』【】《》〈〉〔〕]", "", text)
    text = re.sub(r"[—–―~]", "", text)
    text = text.replace("#", "")
    text = re.sub(r"[※★♪]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "SKIP_EMPTY", None
    return "KEEP", text


def clean_jsonl(jsonl_path: Path, wav_dir: Path) -> None:
    wav_set = {p.name for p in wav_dir.glob("*.wav")}

    kept_records: list[dict] = []
    dropped_lines: list[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped_lines.append(f"line={lineno}  # JSON_DECODE_ERROR")
                continue

            missing = REQUIRED_KEYS - set(rec.keys())
            if missing:
                dropped_lines.append(f"line={lineno}  # MISSING_KEYS:{sorted(missing)}")
                continue

            wav_name = Path(str(rec["audio_path"])).name
            if wav_name not in wav_set:
                dropped_lines.append(f"line={lineno}  # WAV_NOT_FOUND:{rec['audio_path']}")
                continue

            status, cleaned_text = filter_and_clean_tts_text(str(rec["text"]))
            if status != "KEEP":
                dropped_lines.append(f"line={lineno}  # TEXT_{status}")
                continue

            rec["audio_path"] = str(Path(rec["audio_path"]))
            rec["speaker"] = str(rec["speaker"])
            rec["language"] = str(rec["language"])
            rec["text"] = cleaned_text
            kept_records.append(rec)

    backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    jsonl_path.rename(backup_path)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    drop_log = jsonl_path.with_suffix(".dropped.txt")
    with open(drop_log, "w", encoding="utf-8") as f:
        for line in dropped_lines:
            f.write(line + "\n")

    print(f"[{jsonl_path.name}] cleaned")
    print(f"  kept: {len(kept_records)}")
    print(f"  dropped: {len(dropped_lines)}")
    print(f"  backup: {backup_path.name}")
    print(f"  dropped log: {drop_log.name}")


clean_jsonl(META_PATH, WAV_DIR)

