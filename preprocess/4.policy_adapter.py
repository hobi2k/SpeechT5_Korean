from pathlib import Path
from typing import Optional, Tuple
import json
import re

# 원본 데이터 루트.
DATA_ROOT = Path("path/to/data")
# 검증 대상 wav 디렉터리.
WAV_DIR = DATA_ROOT / "sp5"
# filelist 디렉터리.
FILELIST_DIR = DATA_ROOT / "sp5file"
# 3.make_filelists.py에서 생성된 메타 JSONL.
META_PATH = FILELIST_DIR / "metadata.jsonl"

# JSONL에서 반드시 존재해야 하는 키.
REQUIRED_KEYS = {"audio_path", "speaker", "language", "text"}


def filter_and_clean_tts_text(text: str) -> Tuple[str, Optional[str]]:
    """
    TTS 학습 전 텍스트를 정책에 맞춰 정제/필터링한다.

    주요 정책:
    1. 중괄호 주석/태그가 포함된 문장은 드롭
    2. 발화에 불필요한 괄호/특수기호 제거
    3. 공백 정리 후 빈 문자열이면 제외

    Args:
        text: 원본 텍스트.

    Returns:
        Tuple[str, Optional[str]]:
        - status: KEEP / DROP / SKIP_EMPTY
        - cleaned_text: KEEP일 때 정제된 텍스트, 아니면 None
    """
    # { ... } 패턴은 스크립트/주석성 데이터로 간주해 제거.
    if re.search(r"\{[^}]+\}", text):
        return "DROP", None

    # 태그/괄호/장식 기호 제거.
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[「」『』【】《》〈〉〔〕]", "", text)
    text = re.sub(r"[—–―~]", "", text)
    text = text.replace("#", "")
    text = re.sub(r"[※★♪]", "", text)
    # 다중 공백 정리.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "SKIP_EMPTY", None
    return "KEEP", text


def clean_jsonl(jsonl_path: Path, wav_dir: Path) -> None:
    """
    JSONL 메타데이터를 정책 기반으로 정제하고 결과를 원본 경로에 재저장한다.

    처리 순서:
    1. 라인별 JSON 파싱
    2. 필수 키 검증
    3. wav 존재 여부 검증
    4. 텍스트 정책 필터/정제
    5. 통과 레코드만 새 JSONL 작성
    6. 원본 백업(.bak) 및 드롭 사유 로그(.dropped.txt) 기록

    Args:
        jsonl_path: 정제 대상 JSONL 경로.
        wav_dir: wav 파일 존재 여부 검증용 디렉터리.

    Returns:
        None
    """
    # 빠른 존재 검사를 위해 wav 파일명 집합을 미리 구성.
    wav_set = {p.name for p in wav_dir.glob("*.wav")}

    # 결과 버퍼: 통과 레코드와 드롭 라인 로그.
    kept_records: list[dict] = []
    dropped_lines: list[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            # 공백 라인은 무시.
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # JSON 파싱 실패 라인은 드롭 로그에 기록.
                dropped_lines.append(f"line={lineno}  # JSON_DECODE_ERROR")
                continue

            # 스키마 키 누락 검사.
            missing = REQUIRED_KEYS - set(rec.keys())
            if missing:
                dropped_lines.append(f"line={lineno}  # MISSING_KEYS:{sorted(missing)}")
                continue

            # audio_path는 파일명 기준으로 실제 wav 존재를 검증.
            wav_name = Path(str(rec["audio_path"])).name
            if wav_name not in wav_set:
                dropped_lines.append(f"line={lineno}  # WAV_NOT_FOUND:{rec['audio_path']}")
                continue

            # 정책 기반 텍스트 정제/필터 적용.
            status, cleaned_text = filter_and_clean_tts_text(str(rec["text"]))
            if status != "KEEP":
                dropped_lines.append(f"line={lineno}  # TEXT_{status}")
                continue

            # 타입/값 정규화 후 통과 레코드 적재.
            rec["audio_path"] = str(Path(rec["audio_path"]))
            rec["speaker"] = str(rec["speaker"])
            rec["language"] = str(rec["language"])
            rec["text"] = cleaned_text
            kept_records.append(rec)

    # 원본 백업 후 정제본을 동일 경로에 저장.
    backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    jsonl_path.rename(backup_path)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 드롭된 라인의 사유 로그를 별도 파일로 남긴다.
    drop_log = jsonl_path.with_suffix(".dropped.txt")
    with open(drop_log, "w", encoding="utf-8") as f:
        for line in dropped_lines:
            f.write(line + "\n")

    # 정제 요약 출력.
    print(f"[{jsonl_path.name}] cleaned")
    print(f"  kept: {len(kept_records)}")
    print(f"  dropped: {len(dropped_lines)}")
    print(f"  backup: {backup_path.name}")
    print(f"  dropped log: {drop_log.name}")


# 스크립트 직접 실행 시 메타 파일 정제 수행.
clean_jsonl(META_PATH, WAV_DIR)
