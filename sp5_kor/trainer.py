from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
)

from sp5_kor.text.korean_text_utils import (
    decompose_jamo_with_placeholders,
    inject_tokens_for_training,
)

from .config import TrainingConfig
from .data import load_jsonl_dataset
from .tokenizer import build_tokenizer


def _seed_all(seed: int) -> None:
    """
    재현성을 위해 난수 시드를 고정한다.

    Args:
        seed: 고정할 시드 값.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_max_frames(processor: SpeechT5Processor, max_audio_len: int) -> int:
    """
    최대 오디오 길이(초)로부터 mel frame 상한을 계산한다.

    계산식:
        max_frames = ceil(max_audio_len * sampling_rate / hop_length)

    Args:
        processor: SpeechT5Processor. feature_extractor의 sr/hop_length를 사용한다.
        max_audio_len: 최대 오디오 길이(초).

    Returns:
        int: mel frame 상한.
    """
    fe = processor.feature_extractor
    return int(np.ceil(max_audio_len * (fe.sampling_rate / fe.hop_length)))


def _resolve_max_frames(processor: SpeechT5Processor, model: SpeechT5ForTextToSpeech, max_audio_len: int) -> int:
    """
    계산된 max_frames를 모델 positional embedding 한도에 맞게 보정한다.

    Args:
        processor: SpeechT5Processor. frame 계산에 필요한 sr/hop_length를 제공한다.
        model: SpeechT5 모델. `config.max_speech_positions` 한도를 참조한다.
        max_audio_len: 사용자 설정 최대 오디오 길이(초).

    Returns:
        int: 모델이 처리 가능한 최종 max_frames.
    """
    computed = _compute_max_frames(processor, max_audio_len)
    model_limit = int(getattr(model.config, "max_speech_positions", computed))
    resolved = min(computed, model_limit)
    if resolved < computed:
        print(
            f"[WARN] MAX_FRAMES {computed} -> {resolved} "
            f"(clamped by model max_speech_positions={model_limit})"
        )
    return resolved


def _build_speaker_embedding(cfg: TrainingConfig, dataset, device: torch.device) -> torch.Tensor:
    """
    단일 화자 기준 reference speaker embedding을 생성한다.

    첫 번째 샘플의 waveform을 이용해 WavLM X-Vector를 추출하고 정규화한다.

    Args:
        cfg: 학습 설정. target_sr, speaker model 이름을 사용한다.
        dataset: waveform 컬럼을 포함한 데이터셋.
        device: speaker model을 올릴 디바이스(CPU 권장).

    Returns:
        torch.Tensor: shape=(emb_dim,)인 정규화된 speaker embedding (CPU 텐서).
    """
    spk_model = WavLMForXVector.from_pretrained(cfg.speaker_model_name).to(device).eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg.speaker_model_name)

    waveform = np.asarray(dataset[0]["waveform"], dtype=np.float32)
    inputs = fe(waveform, sampling_rate=cfg.target_sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        emb = spk_model(**inputs).embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb.squeeze().cpu()


def _ensure_local_snapshot(repo_id: str, local_dir: Path) -> Path:
    """
    Hugging Face 모델 스냅샷을 로컬 폴더에 보장한다.

    이미 로컬에 있으면 재다운로드하지 않는다.

    Args:
        repo_id: Hugging Face repo id.
        local_dir: 로컬 저장 디렉터리.

    Returns:
        Path: 로컬 스냅샷 디렉터리 경로.
    """
    if (local_dir / "config.json").exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


@dataclass
class _Collator:
    """
    TTS 학습용 배치 콜레이터.

    Attributes:
        pad_token_id: tokenizer의 pad token id.
        spk_emb: 단일 화자 embedding. 배치 크기만큼 복제해 사용한다.
    """

    pad_token_id: int
    spk_emb: torch.Tensor

    def __call__(self, features):
        """
        가변 길이 샘플 목록을 모델 입력 배치 텐서로 변환한다.

        Args:
            features: 데이터셋 샘플 목록. 각 샘플은 input_ids/labels를 포함한다.

        Returns:
            dict: 모델 forward 입력 딕셔너리.
            - input_ids: LongTensor [B, T]
            - attention_mask: LongTensor [B, T]
            - labels: FloatTensor [B, F, M]
            - speaker_embeddings: FloatTensor [B, emb_dim]
        """
        # input_ids를 배치 내 최대 길이에 맞춰 padding.
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        # pad 위치를 제외한 attention mask 생성.
        attention_mask = (input_ids != self.pad_token_id).long()

        # mel labels 스택.
        labels = torch.stack([torch.tensor(f["labels"], dtype=torch.float32) for f in features])

        # 0.0 패딩 프레임을 손실 제외 값(-100)으로 치환.
        labels = labels.masked_fill(labels.eq(0.0), -100.0)

        # 단일 화자 embedding을 배치 크기만큼 복제.
        spk = self.spk_emb.unsqueeze(0).repeat(len(features), 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "speaker_embeddings": spk,
        }


def run_training(cfg: TrainingConfig) -> Path:
    """
    JSONL 기반 SpeechT5 학습을 end-to-end로 실행한다.

    파이프라인:
    1. 시드 고정, 출력 디렉터리 준비
    2. tokenizer / dataset / model / vocoder 로드
    3. speaker embedding 생성
    4. 고정 아티팩트(config, generation_config, tokenizer, vocoder, utils, spk) 선저장
    5. alignment 안전 필터(audio_sec, token_len, token_per_sec)
    6. 전처리(mel 생성) 후 학습/검증 루프 실행
    7. best checkpoint 로드 후 모델 가중치 저장

    Args:
        cfg: 학습 설정 객체.

    Returns:
        Path: 최종 아티팩트가 저장된 출력 디렉터리 경로.

    Raises:
        RuntimeError: 필터 후 데이터셋이 비어 학습이 불가능한 경우.
    """
    _seed_all(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) tokenizer 준비
    tokenizer = build_tokenizer(cfg.jamo_vocab_path, cfg.output_dir)
    print("Tokenizer vocab size:", len(tokenizer))

    # 2) 데이터셋 로드
    ds = load_jsonl_dataset(cfg)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty after audio filtering.")

    # 3) 사전학습 모델을 프로젝트 내부 pretrained 폴더에 보장한다.
    # 항상 프로젝트 루트의 pretrained 디렉터리를 사용한다.
    pretrained_root = Path(__file__).resolve().parent.parent / "pretrained"
    tts_local = _ensure_local_snapshot(cfg.model_name, pretrained_root / "speecht5_tts")
    vocoder_local = _ensure_local_snapshot(cfg.vocoder_name, pretrained_root / "speecht5_hifigan")
    spk_local = _ensure_local_snapshot(cfg.speaker_model_name, pretrained_root / "wavlm_base_plus_sv")
    print(f"Using local pretrained assets from: {pretrained_root}")

    # speaker embedding 함수는 cfg.speaker_model_name을 참조하므로 로컬 경로로 치환한다.
    cfg.speaker_model_name = str(spk_local)

    # 4) 모델/프로세서 로드 (로컬 pretrained 경로 기반)
    processor = SpeechT5Processor.from_pretrained(str(tts_local))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(tts_local))
    vocoder = SpeechT5HifiGan.from_pretrained(str(vocoder_local))

    # 커스텀 tokenizer를 SpeechT5 processor/model에 연결.
    processor.tokenizer = tokenizer
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    model.config.use_guided_attn_loss = False

    model.to(cfg.device)
    vocoder.to(cfg.device)

    # mel frame 상한 계산.
    max_frames = _resolve_max_frames(processor, model, cfg.max_audio_len)
    print("Resolved MAX_FRAMES:", max_frames)

    # 4) speaker embedding (CPU 고정 권장).
    ref_spk_emb = _build_speaker_embedding(cfg, ds, torch.device("cpu"))
    print("Speaker embedding shape:", tuple(ref_spk_emb.shape))

    # 5) 학습 전에 고정 아티팩트 저장.
    # 중간 중단 시에도 로컬 추론 smoke test를 할 수 있게 한다.
    model.config.speaker_embedding_path = "speaker_embedding.pth"
    model.config.save_pretrained(cfg.output_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    vocoder.save_pretrained(cfg.output_dir / "vocoder")
    torch.save(ref_spk_emb, cfg.output_dir / "speaker_embedding.pth")

    utils_src = Path(__file__).resolve().parent / "text" / "korean_text_utils.py"
    if utils_src.exists():
        shutil.copy2(utils_src, cfg.output_dir / "korean_text_utils.py")

    print(f"Saved fixed artifacts (config/tokenizer/vocoder/utils/spk) to: {cfg.output_dir}")

    def _alignment_guard(batch):
        """
        샘플의 정렬 안정성을 보장하기 위한 통계/필터 메타를 생성한다.

        Args:
            batch: 단일 샘플(dict). text/waveform을 포함한다.

        Returns:
            dict: keep/drop 판단 및 로그 통계를 담은 메타 필드.
            - _keep: 유지 여부
            - _drop_reason: 제거 사유 문자열
            - _audio_sec: 오디오 길이(초)
            - _token_len: 자모 토큰 길이
            - _token_per_sec: token_len / audio_sec
        """
        txt = (batch.get("text") or "").strip()
        waveform = batch.get("waveform") or []

        audio_sec = len(waveform) / float(cfg.target_sr) if len(waveform) > 0 else 0.0
        if not txt:
            return {
                "_keep": False,
                "_drop_reason": "empty_text",
                "_audio_sec": audio_sec,
                "_token_len": 0,
                "_token_per_sec": 0.0,
            }

        safe_text = inject_tokens_for_training(txt)
        jamo_seq = decompose_jamo_with_placeholders(safe_text)
        token_len = len(jamo_seq)
        token_per_sec = token_len / audio_sec if audio_sec > 0 else 0.0

        drop_reason = ""
        if audio_sec < cfg.min_audio_len:
            drop_reason = "audio_too_short"
        elif token_len < cfg.min_tokens:
            drop_reason = "token_too_short"
        elif token_len > cfg.max_tokens:
            drop_reason = "token_too_long"
        elif token_per_sec < cfg.min_token_per_sec:
            drop_reason = "token_per_sec_too_low"
        elif token_per_sec > cfg.max_token_per_sec:
            drop_reason = "token_per_sec_too_high"

        return {
            "_keep": drop_reason == "",
            "_drop_reason": drop_reason if drop_reason else "keep",
            "_audio_sec": float(audio_sec),
            "_token_len": int(token_len),
            "_token_per_sec": float(token_per_sec),
        }

    # 6) alignment 필터 메타 생성.
    ds = ds.map(_alignment_guard, num_proc=max(cfg.num_proc, 1))

    # 7) 필터 로그 출력.
    total_before = len(ds)
    reasons = Counter(ds["_drop_reason"])
    kept_before_filter = reasons.get("keep", 0)
    print(
        "[Filter] total="
        f"{total_before} kept={kept_before_filter} dropped={total_before - kept_before_filter}"
    )
    for reason, cnt in sorted(reasons.items()):
        if reason == "keep":
            continue
        print(f"[Filter] drop_reason={reason} count={cnt}")

    kept_audio_sec = [a for a, k in zip(ds["_audio_sec"], ds["_keep"]) if k]
    kept_token_len = [t for t, k in zip(ds["_token_len"], ds["_keep"]) if k]
    kept_tps = [x for x, k in zip(ds["_token_per_sec"], ds["_keep"]) if k]
    if kept_audio_sec:
        print(
            "[Filter] kept_audio_sec "
            f"min={min(kept_audio_sec):.2f} p50={np.percentile(kept_audio_sec, 50):.2f} "
            f"p95={np.percentile(kept_audio_sec, 95):.2f} max={max(kept_audio_sec):.2f}"
        )
        print(
            "[Filter] kept_token_len "
            f"min={min(kept_token_len)} p50={int(np.percentile(kept_token_len, 50))} "
            f"p95={int(np.percentile(kept_token_len, 95))} max={max(kept_token_len)}"
        )
        print(
            "[Filter] kept_token_per_sec "
            f"min={min(kept_tps):.2f} p50={np.percentile(kept_tps, 50):.2f} "
            f"p95={np.percentile(kept_tps, 95):.2f} max={max(kept_tps):.2f}"
        )

    # 8) keep 샘플만 남기고 임시 메타 컬럼 제거.
    ds = ds.filter(lambda x: x["_keep"])
    ds = ds.remove_columns(["_keep", "_drop_reason", "_audio_sec", "_token_len", "_token_per_sec"])
    if len(ds) == 0:
        raise RuntimeError("Dataset became empty after text filtering.")

    def _preprocess(batch):
        """
        모델 입력용 텍스트/오디오 전처리를 수행한다.

        Args:
            batch: 단일 샘플(dict). text/waveform을 포함한다.

        Returns:
            dict: 모델 forward 입력에 필요한 3개 필드.
            - input_ids
            - attention_mask
            - labels (mel target)
        """
        safe_text = inject_tokens_for_training(batch["text"])
        jamo_seq = decompose_jamo_with_placeholders(safe_text)

        encoded = tokenizer(
            jamo_seq,
            is_split_into_words=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        # truncate 없이 원본 mel 길이를 먼저 계산한다.
        mel = processor(
            audio_target=batch["waveform"],
            sampling_rate=cfg.target_sr,
        )["input_values"][0].astype(np.float32)

        # 모델 허용 프레임을 넘는 샘플은 드롭 대상으로 표시한다.
        if mel.shape[0] > max_frames:
            n_mels = mel.shape[1]
            return {
                "_drop_preprocess": True,
                "_drop_preprocess_reason": "frame_too_long",
                "input_ids": [],
                "attention_mask": [],
                "labels": np.zeros((1, n_mels), dtype=np.float32),
            }

        # 허용 길이보다 짧으면 우측 패딩한다. (truncate는 하지 않음)
        if mel.shape[0] < max_frames:
            n_mels = mel.shape[1]
            pad = np.zeros((max_frames - mel.shape[0], n_mels), dtype=np.float32)
            mel = np.concatenate([mel, pad], axis=0)

        return {
            "_drop_preprocess": False,
            "_drop_preprocess_reason": "keep",
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": mel,
        }

    # 9) 학습 입력셋 생성.
    model_ds = ds.map(_preprocess, remove_columns=ds.column_names)

    # 9-1) preprocess 단계 frame 필터 로그 출력.
    pre_reasons = Counter(model_ds["_drop_preprocess_reason"])
    pre_kept = pre_reasons.get("keep", 0)
    print(
        "[PreprocessFilter] total="
        f"{len(model_ds)} kept={pre_kept} dropped={len(model_ds) - pre_kept}"
    )
    for reason, cnt in sorted(pre_reasons.items()):
        if reason == "keep":
            continue
        print(f"[PreprocessFilter] drop_reason={reason} count={cnt}")

    # keep 샘플만 남기고 임시 메타를 제거한다.
    model_ds = model_ds.filter(lambda x: not x["_drop_preprocess"])
    model_ds = model_ds.remove_columns(["_drop_preprocess", "_drop_preprocess_reason"])
    if len(model_ds) == 0:
        raise RuntimeError("Dataset became empty after preprocess frame filter.")

    # 10) train/test 분할.
    split = int(len(model_ds) * cfg.train_ratio)
    split = min(max(split, 1), len(model_ds) - 1) if len(model_ds) > 1 else 1
    if len(model_ds) == 1:
        tts = DatasetDict({"train": model_ds, "test": model_ds})
    else:
        tts = DatasetDict(
            {
                "train": model_ds.select(range(split)),
                "test": model_ds.select(range(split, len(model_ds))),
            }
        )
    print("Train size:", len(tts["train"]), "| Test size:", len(tts["test"]))

    # 11) DataLoader 구성.
    collator = _Collator(pad_token_id=tokenizer.pad_token_id, spk_emb=ref_spk_emb)
    train_dl = DataLoader(tts["train"], batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(tts["test"], batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)

    # 12) optimizer / scaler 준비.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(cfg.device.type == "cuda"))

    last_ckpt = cfg.output_dir / "checkpoint_last.pt"
    periodic_dir = cfg.output_dir / "checkpoints"
    periodic_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if cfg.resume:
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=cfg.device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler_state = ckpt.get("scaler")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(
                f"[Resume] loaded {last_ckpt} | "
                f"last_epoch={start_epoch - 1} next_epoch={start_epoch}"
            )
        else:
            print(f"[Resume] checkpoint not found: {last_ckpt}. Start from epoch 1.")

    # 13) 학습/검증 루프.
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        tr_total = 0.0
        for b in tqdm(train_dl, desc=f"Train {epoch}"):
            b = {k: v.to(cfg.device) for k, v in b.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=cfg.device.type, enabled=(cfg.device.type == "cuda")):
                out = model(**b)

            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_total += float(out.loss.item())

        tr_loss = tr_total / max(len(train_dl), 1)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for b in tqdm(val_dl, desc=f"Eval {epoch}"):
                b = {k: v.to(cfg.device) for k, v in b.items()}
                out = model(**b)
                val_total += float(out.loss.item())

        val_loss = val_total / max(len(val_dl), 1)
        print(f"[{epoch}] train={tr_loss:.4f} | val={val_loss:.4f}")

        # 주기 저장: save_every_n_epochs마다 HF 모델 디렉터리를 저장한다.
        if cfg.save_every_n_epochs > 0 and (epoch % cfg.save_every_n_epochs == 0):
            periodic_model_dir = periodic_dir / f"epoch_{epoch:06d}"
            periodic_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(periodic_model_dir)
            print(f"  -> Periodic model saved to {periodic_model_dir}")

            # 최대 보관 개수를 넘기면 가장 오래된 주기 모델 디렉터리부터 삭제한다.
            periodic_models = sorted(
                p for p in periodic_dir.glob("epoch_*") if p.is_dir()
            )
            overflow = len(periodic_models) - cfg.max_periodic_checkpoints
            if overflow > 0:
                for old_model_dir in periodic_models[:overflow]:
                    shutil.rmtree(old_model_dir, ignore_errors=True)
                    print(f"  -> Removed old periodic model: {old_model_dir}")

        # 매 epoch 마지막 상태를 저장하여 언제든 이어서 학습 가능하게 한다.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if cfg.device.type == "cuda" else None,
                "epoch": epoch,
                "best": None,
            },
            last_ckpt,
        )
        print(f"  -> Last checkpoint saved to {last_ckpt}")

    # 14) 마지막 epoch 상태를 최종 모델 아티팩트로 저장.
    model.to(cfg.device).eval()

    # model artifacts (after training)
    model.save_pretrained(cfg.output_dir)
    print(f"Saved model weights to: {cfg.output_dir}")
    return cfg.output_dir
