"""
sp5_kor public API.

Exports:
- TrainingConfig: 학습 설정 dataclass
- run_training: JSONL 기반 학습 엔트리 함수
"""

from .config import TrainingConfig
from .trainer import run_training

__all__ = ["TrainingConfig", "run_training"]
