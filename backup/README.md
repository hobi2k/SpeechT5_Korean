# SpeechT5_Korean

한국어 TTS를 위해 `SpeechT5`를 자모(Jamo) 기반 토크나이저로 학습/추론하는 프로젝트입니다.

현재 기준 메인 실행 파일은 단일 스크립트 기반입니다.
- 학습: `train_speecht5.py`
- 추론: `inference.py`
- 자모 vocab 생성: `jamo_vocab_builder.py`
- 텍스트 전처리 유틸: `korean_text_utils.py`

## 프로젝트 구조

`SpeechT5_Korean/` 기준 주요 파일:
- `train_speecht5.py`: KSS 데이터셋 로드, 전처리, 학습, 체크포인트 저장, Hub 업로드
- `inference.py`: Hub에서 모델/토크나이저/speaker embedding 로드 후 음성 합성
- `jamo_vocab_builder.py`: `jamo_vocab.txt` 생성
- `korean_text_utils.py`: 숫자/단위/구두점 처리, 자모 분해, prosody 분할 유틸
- `jamo_vocab.txt`: 학습에 사용하는 자모 vocab (이미 생성되어 있음)
- `requirements.txt`, `pyproject.toml`, `uv.lock`: 의존성 관리
- `train_speecht5_backup.py`, `train_speecht5_test.py`, `backup/`: 실험/백업 파일
- `spt5_kor/`: 현재 빈 패키지(미사용)

## 현재 동작 요약

`train_speecht5.py`는 아래를 수행합니다.
1. `jamo_vocab.txt`를 로드해 `PreTrainedTokenizerFast` 구성
2. `Bingsu/KSS_Dataset` 로드 후 오디오 리샘플링/길이 필터링
3. WavLM 기반 단일 화자 speaker embedding 추출
4. 텍스트를 placeholder + 자모 시퀀스로 변환
5. SpeechT5 학습 후 best checkpoint 저장
6. 모델/토크나이저/vocoder/speaker embedding 저장
7. 모델 카드 및 demo 스크립트 생성
8. Hugging Face Hub로 업로드

주의:
- 학습 스크립트는 마지막에 `create_and_push_to_hub()`를 호출합니다.
- Hub 업로드를 원치 않으면 스크립트 수정이 필요합니다.

## 환경 준비

Python:
- `>=3.11`

의존성 설치(`uv`):
```bash
cd SpeechT5_Korean
uv sync --prerelease=allow
```

PyTorch (RTX 50 계열 권장):
- 현재 의존성은 nightly/cu128 계열을 허용하도록 설정되어 있습니다.
- 예시:
```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

추가 참고:
- `datasets`의 `Audio(decode=True)` 사용 시 `torchcodec`가 필요할 수 있습니다.
- 시스템 FFmpeg/torchcodec/PyTorch 조합이 맞지 않으면 audio decode 단계에서 실패할 수 있습니다.

## 실행 방법

### 1) 자모 vocab 생성(필요 시)
```bash
cd SpeechT5_Korean
uv run jamo_vocab_builder.py
```

### 2) 학습
```bash
cd SpeechT5_Korean
uv run train_speecht5.py
```

학습 스킵(아티팩트 저장/업로드 파이프라인 점검):
```bash
cd SpeechT5_Korean
uv run train_speecht5.py --skip_train
```

### 3) 추론
```bash
cd SpeechT5_Korean
uv run inference.py
```

## 데이터셋

현재 학습 코드는 아래 데이터셋을 하드코딩해 사용합니다.
- `Bingsu/KSS_Dataset` (Hugging Face Datasets)

로컬 JSONL 학습 파이프라인은 아직 메인 스크립트에 반영되지 않았습니다.

## 출력 아티팩트

기본 저장 경로:
- `speecht5_kss_korean_jamo/`

주요 산출물:
- 학습 모델 가중치/설정
- `tokenizer.json` 및 tokenizer 관련 파일
- `vocoder/`
- `speaker_embedding.pth`
- `korean_text_utils.py` 복사본
- `README.md`(자동 생성 모델 카드)
- `demo_inference.py`