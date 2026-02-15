# SpeechT5_Korean

본 프로젝트는 SpeechT5 한글화 프로젝트입니다.
TTS 학습과 실험용으로 제작되었습니다.

현재 메인 엔트리:
- 학습: `scripts/train.py`
- 추론: `scripts/inference.py`
- 텍스트/자모 유틸: `sp5_kor/text/`

## 프로젝트 구조

핵심 디렉터리:
- `sp5_kor/config.py`: 학습 설정 dataclass
- `sp5_kor/data.py`: JSONL 로드 + 오디오 로딩/리샘플/필터링
- `sp5_kor/tokenizer.py`: `jamo_vocab.txt` 기반 tokenizer 생성
- `sp5_kor/trainer.py`: 전처리/학습/검증/아티팩트 저장
- `sp5_kor/text/jamo_vocab_builder.py`: 자모 vocab 생성
- `sp5_kor/text/korean_text_utils.py`: 텍스트 정규화/placeholder/자모 분해
- `scripts/train.py`: JSONL 학습 CLI
- `scripts/inference.py`: 로컬 모델 추론 CLI
- `pretrained/`: Hugging Face 사전학습 모델 로컬 캐시 디렉터리

## JSONL 데이터셋 포맷

각 라인은 아래 키를 포함해야 합니다.
- `audio_path`
- `speaker`
- `language`
- `text`

예시:
```json
{"audio_path":"wavs/0001.wav","speaker":"spk1","language":"ko","text":"안녕하세요"}
```

`audio_path` 해석 규칙:
- 절대경로면 그대로 사용
- 상대경로면 JSONL 파일이 있는 디렉터리 기준으로 해석

## 환경 준비

```bash
cd SpeechT5_Korean
uv sync
```

RTX 50 계열이면 torch nightly/cu128 권장

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 실행

1. 학습 실행

```bash
cd SpeechT5_Korean
uv run scripts/train.py \
  --jsonl_path /path/to/train.jsonl \
  --output_dir /path/to/output_model \
  --min_audio_len 0.8 \
  --max_audio_len 10 \
  --min_tokens 5 \
  --max_tokens 200 \
  --min_token_per_sec 2.0 \
  --max_token_per_sec 35.0
```

중단 후 이어서 학습

```bash
uv run scripts/train.py \
  --jsonl_path /path/to/train.jsonl \
  --output_dir /path/to/output_model \
  --num_epochs 80 \
  --resume
```

주의:
- `--num_epochs`는 \"추가 에폭\"이 아니라 \"최종 목표 epoch\"입니다.
- 예: 이전에 50 epoch까지 끝났고 `--num_epochs 80 --resume`이면 51~80만 추가로 학습합니다.
- 사전학습 모델(`speecht5_tts`, `speecht5_hifigan`, `wavlm-base-plus-sv`)은 첫 실행 시 프로젝트 루트 `pretrained/` 아래로 다운로드되고 이후 재사용됩니다.
- 중간 저장 모델은 `checkpoints/epoch_XXXXXX/`(HF 모델 디렉터리) 형식으로 보관되며, 추론 시 `--checkpoint_epoch`로 선택할 수 있습니다.

학습 시 필터 로그가 출력됩니다.
- 전체/유지/제거 샘플 수
- 제거 사유별 카운트
- 유지 샘플 분포(`audio_sec`, `token_len`, `token_per_sec`의 min/p50/p95/max)

2. 로컬 추론

```bash
cd SpeechT5_Korean
uv run scripts/inference.py \
  --model_dir /path/to/output_model \
  --text "안녕하세요. 로컬 추론 테스트입니다." \
  --out out.wav
```

10 epoch 주기 저장 모델 중 특정 epoch를 선택해 추론할 수 있습니다.

```bash
uv run scripts/inference.py \
  --model_dir /path/to/output_model \
  --checkpoint_epoch 40 \
  --text "40 epoch 체크포인트 모델로 추론합니다." \
  --out out_epoch40.wav
```

추론 경로를 직접 지정할 수도 있습니다.

```bash
uv run scripts/inference.py \
  --model_dir /path/to/output_model \
  --text_utils_path /path/to/output_model/korean_text_utils.py \
  --vocoder_dir /path/to/output_model/vocoder \
  --speaker_embedding_path /path/to/output_model/speaker_embedding.pth
```

## 저장 아티팩트

`--output_dir` 저장 타이밍:
- 학습 시작 전(고정 아티팩트): `config/generation_config/tokenizer/vocoder/utils/speaker_embedding`
- 학습 완료 후: 모델 가중치

생성 파일:
- 모델 파일 (`model.safetensors` 등)
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `korean_text_utils.py`
- `speaker_embedding.pth`
- `vocoder/`
- `checkpoint_last.pt` (매 epoch 마지막 상태: model/optimizer/scaler/epoch)
- `checkpoints/epoch_XXXXXX/` (10 epoch 단위 HF 모델 저장, 최대 5개 유지)

## 인용

이 프로젝트가 유용했다면 아래 형식으로 인용해 주세요.

```bibtex
@misc{speecht5_korean,
  title        = {SpeechT5_Korean: Korean SpeechT5 Training and Inference Pipeline},
  author       = {안호성 (GitHub: hobi2k)},
  year         = {2026},
  url          = {https://github.com/hobi2k/SpeechT5_Korean},
  note         = {Hugging Face: https://huggingface.co/ahnhs2k, Accessed: 2026-02-15}
}
```

## 참고 및 크레딧

- Microsoft SpeechT5: https://github.com/microsoft/SpeechT5
- Hugging Face `microsoft/speecht5_tts`: https://huggingface.co/microsoft/speecht5_tts
- Hugging Face `microsoft/speecht5_hifigan`: https://huggingface.co/microsoft/speecht5_hifigan
- Hugging Face `microsoft/wavlm-base-plus-sv`: https://huggingface.co/microsoft/wavlm-base-plus-sv
