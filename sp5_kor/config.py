from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainingConfig:
    """
    SpeechT5 Korean JSONL 학습 설정.

    이 설정은 데이터 로딩, 정렬 안정성 필터링, 학습 하이퍼파라미터,
    모델/보코더 이름, 실행 디바이스까지 포함한다.
    """

    # Dataset / path options
    # JSONL 파일 경로 (필수).
    jsonl_path: Path
    # 학습 결과 아티팩트를 저장할 출력 디렉터리 (필수).
    output_dir: Path
    # 자모 vocab 파일 경로.
    jamo_vocab_path: Path = Path("./sp5_kor/text/jamo_vocab.txt")

    # Audio / text filtering options
    # 목표 샘플링레이트.
    target_sr: int = 16000
    # 허용 최소 오디오 길이(초).
    min_audio_len: float = 0.8
    # 허용 최대 오디오 길이(초). 초과 샘플은 drop.
    max_audio_len: int = 10
    # 허용 최소 토큰 길이.
    min_tokens: int = 5
    # 허용 최대 토큰 길이.
    max_tokens: int = 200
    # 허용 최소 token/sec 비율.
    min_token_per_sec: float = 2.0
    # 허용 최대 token/sec 비율.
    max_token_per_sec: float = 35.0
    # train/test 분할 비율.
    train_ratio: float = 0.98
    # datasets.map/filter 병렬 처리 프로세스 수.
    num_proc: int = 4

    # Optim / training options
    # 배치 크기.
    batch_size: int = 3
    # 학습 epoch 수.
    num_epochs: int = 100
    # output_dir/checkpoint_last.pt 기준으로 학습을 이어갈지 여부.
    resume: bool = False
    # AdamW 학습률.
    lr: float = 1e-4
    # AdamW weight decay.
    weight_decay: float = 1e-6
    # 랜덤 시드.
    seed: int = 42

    # Base model options
    # SpeechT5 기반 모델 ID.
    model_name: str = "microsoft/speecht5_tts"
    # Vocoder 모델 ID.
    vocoder_name: str = "microsoft/speecht5_hifigan"
    # 화자 임베딩 추출 모델 ID.
    speaker_model_name: str = "microsoft/wavlm-base-plus-sv"

    # 실행 디바이스.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
