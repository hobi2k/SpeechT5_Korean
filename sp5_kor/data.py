from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Dataset, load_dataset

from .config import TrainingConfig


REQUIRED_JSONL_COLUMNS = {"audio_path", "speaker", "language", "text"}


def load_jsonl_dataset(cfg: TrainingConfig) -> Dataset:
    """
    JSONL 학습 메타데이터를 로드하고 오디오를 파형으로 변환해 반환한다.

    입력 JSONL 스키마는 다음 키를 반드시 포함해야 한다.
    - `audio_path`
    - `speaker`
    - `language`
    - `text`

    처리 단계:
    1. JSONL 로드 및 필수 컬럼 검증
    2. `audio_path`를 절대 경로로 해석
    3. 오디오 파일 읽기(`soundfile`)
    4. 모노 변환 / float32 변환 / target_sr 리샘플
    5. `max_audio_len` 초과 샘플 drop 처리

    Args:
        cfg: 학습 설정 객체. 경로, 샘플링레이트, 최대 길이, 병렬 처리 수를 사용한다.

    Returns:
        Dataset: 오디오가 로드되어 `waveform(list[float])`, `sr(int)`가 추가된 데이터셋.
        길이 초과 또는 파일 누락 샘플은 제거된 상태로 반환한다.

    Raises:
        ValueError: JSONL 필수 컬럼이 누락된 경우.
    """
    # JSONL 파일을 train split 하나로 로드한다.
    ds = load_dataset("json", data_files={"train": str(cfg.jsonl_path)})["train"]

    # 필수 스키마 검증.
    missing = REQUIRED_JSONL_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(f"Missing required JSONL columns: {sorted(missing)}")

    # 상대 경로 audio_path의 기준 디렉터리는 항상 jsonl 파일의 부모 경로다.
    base_dir = cfg.jsonl_path.parent

    def _resolve_and_load(batch):
        """
        단일 샘플의 audio_path를 실제 파형으로 변환한다.

        Args:
            batch: datasets.map이 넘겨주는 샘플(dict).

        Returns:
            dict: 원본 샘플에 아래 필드를 추가/수정한 결과.
            - `audio_path`: 절대/해석된 경로 문자열
            - `waveform`: float32 1D 파형(list)
            - `sr`: 샘플링레이트(int)
            - `drop`: 제거 여부(bool)
        """
        raw_path = Path(batch["audio_path"])

        # audio_path가 상대경로면 base_dir 기준으로 해석한다.
        wav_path = raw_path if raw_path.is_absolute() else (base_dir / raw_path)

        # 파일이 없으면 샘플을 drop 처리한다.
        if not wav_path.exists():
            batch["drop"] = True
            batch["waveform"] = []
            batch["sr"] = cfg.target_sr
            return batch

        # 오디오 로드.
        arr, sr = sf.read(str(wav_path))

        # 채널 차원 정규화.
        arr = np.asarray(arr)
        if arr.ndim == 2:
            # stereo/multi-channel -> mono 평균.
            arr = arr.mean(axis=-1)

        # 모델 입력 일관성을 위해 float32 1D로 고정한다.
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)

        # 샘플레이트가 다르면 target_sr로 리샘플한다.
        if sr != cfg.target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=cfg.target_sr).astype(np.float32)
            sr = cfg.target_sr

        # 길이 초과 샘플은 자르지 않고 drop 처리한다.
        max_len = int(cfg.max_audio_len * sr)
        too_long = len(arr) > max_len

        batch["drop"] = bool(too_long)
        batch["waveform"] = [] if too_long else arr.tolist()
        batch["sr"] = int(sr)
        batch["audio_path"] = str(wav_path)
        return batch

    # 샘플 단위 오디오 로딩/정규화.
    ds = ds.map(_resolve_and_load, num_proc=max(cfg.num_proc, 1))

    # drop 플래그가 True인 샘플 제거.
    ds = ds.filter(lambda x: not x.get("drop", False))
    return ds
