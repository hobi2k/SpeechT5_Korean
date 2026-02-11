# Hugging Face datasets 라이브러리에서 데이터셋 로드 및 Audio 처리 함수 import
from datasets import load_dataset, Audio

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 최상위 저장 경로를 Path 객체로 정의
OUT_DIR = Path("/mnt/d/tts_data/yae_ko")
# audio 폴더 경로
AUDIO_DIR = OUT_DIR / "audio"
# audio 폴더 생성
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 데이터셋 로드 (streaming)
dataset = load_dataset(
    # 데이터셋 ID (원신 보이스 데이터)
    "simon3000/genshin-voice",
    # train 스플릿만 로드
    split="train",
    # 스트리밍 모드 (전체 다운로드 X, 필요시 다운로드)
    streaming=True,
)

# audio 자동 디코딩
# audio 컬럼을 Audio 타입으로 변환 (decode=False: 바이너리 유지)
dataset = dataset.cast_column("audio", Audio(decode=False))

# 필터
dataset = dataset.filter(
    # 람다 함수 정의 (각 데이터 항목 v에 대해)
    lambda v:
        # 언어가 Korean이고
        v["language"] == "Korean"
        # 화자가 Yae Miko (야에 미코)이고
        and v["speaker"] == "Yae Miko"
        # 전사(자막) 텍스트가 비어있지 않은 항목만
        and v["transcription"] != ""
)

# 메타데이터 저장할 빈 리스트 생성
rows = []
# 파일 인덱스 (파일명용) 초기화
idx = 0

# 필터된 데이터셋 순회 (tqdm으로 진행상황 표시)
for item in tqdm(dataset):
    # wav 파일 경로 생성 (00000.wav, 00001.wav, ...)
    wav_path = AUDIO_DIR / f"{idx:05d}.wav"
    # 데이터 항목에서 오디오 바이트 데이터 추출
    audio_bytes = item["audio"]["bytes"]

    # 오디오 데이터가 없으면 (None이면)
    if audio_bytes is None:
        # 일부 샘플 방어
        # 현재 반복 건너뛰고 다음 항목으로 진행
        continue

    # wav 파일을 바이너리 쓰기 모드로 열기
    with open(wav_path, "wb") as f:
        # 오디오 바이트 데이터를 파일에 작성
        f.write(audio_bytes)

    # 메타데이터 딕셔너리를 rows 리스트에 추가
    rows.append({
        # wav 파일 경로 (Path 객체를 문자열로 변환)
        "wav": str(wav_path),
        # 오디오의 전사 텍스트
        "text": item["transcription"],
    })

    # 인덱스 1 증가
    idx += 1

# 메타데이터 저장
# rows 리스트를 pandas DataFrame으로 변환
df = pd.DataFrame(rows)
# CSV 파일로 저장 (인덱스 제외, UTF-8 인코딩)
df.to_csv(OUT_DIR / "metadata_raw.csv", index=False, encoding="utf-8")

# 완료 메시지 출력 (저장된 샘플 개수 표시)
print(f"[DONE] Saved {len(df)} Korean Jean samples")
