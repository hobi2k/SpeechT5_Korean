"""
SpeechT5 Korean TTS Training Script
(Jamo tokenizer + placeholder 기반, KSS Dataset, HF Hub 업로드)

1. "텍스트를 모델이 처리할 수 있는 토큰 시퀀스"로 만듭니다.
   - 한국어를 완성형(가-힣) 대신 "자모(초/중/종성)"로 분해하여 입력 토큰 수를 줄이고
     OOV를 최소화(원칙적으로 한글은 자모로 모두 표현 가능).
   - 숫자/단위/구두점은 훈련 시 발음 확정을 피하고 placeholder로 유지해
     음성-텍스트 정렬(alignment) 위험을 줄입니다.
     예: 3.5kg -> <num><unit_kg> (훈련)
     추론에서는 normalize_korean을 통해 삼 점 오 킬로그램으로 바꿀 수 있음

2. 오디오를 SpeechT5가 기대하는 방식으로 전처리합니다.
   - SpeechT5Processor(audio_target=...)가 mel target(정확히는 log-mel feature)을 생성
   - 이 mel이 decoder의 "teacher forcing target" 역할을 하며 loss가 계산됩니다.

3. KSS는 단일 화자 데이터셋이므로 speaker embedding을 1개만 뽑아서 고정합니다.
   - WavLM speaker verification 모델로 embedding을 추출합니다.
   - 모델 학습/추론 모두에서 동일 embedding 사용(= 단일 화자 음색 고정)

4. 학습 후 HF Hub에 필요한 아티팩트를 모두 저장/업로드합니다.
   - model, tokenizer, vocoder, speaker_embedding, korean_text_utils.py, README, demo script

주의
- KSS는 텍스트가 상당히 정규화된 편이지만,
  이 스크립트는 "placeholder 기반"이므로 숫자 발음 정보를 일부 버립니다.

"""

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from datasets import load_dataset, Audio, DatasetDict

from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
    PreTrainedTokenizerFast,
)

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from huggingface_hub import HfApi, hf_hub_download

import argparse

# korean_text_utils: 훈련용/추론용 텍스트 유틸
# - inject_tokens_for_training: 훈련 시 숫자/단위/구두점 placeholder 치환
# - decompose_jamo_with_placeholders: placeholder는 통째로 유지하고 나머지 한글만 자모 분해
# - prosody_split / prosody_pause: 추론 데모에서 문장 분할 및 무음 삽입 길이 산정
from korean_text_utils import (
    inject_tokens_for_training,
    decompose_jamo_with_placeholders,
    prosody_split,
    prosody_pause,
)

# CLI (Command Line Interface)
# - 학습을 건너뛰고(훈련 없이) 저장/업로드 파이프라인만 테스트하고 싶을 때 유용
# 사용 예:
#   uv run train.py --skip_train
parser = argparse.ArgumentParser()
parser.add_argument(
    "--skip_train",
    action="store_true",
    help="학습 스킵 (저장/업로드만 테스트하고 싶을 때)",
)
args = parser.parse_args()


@dataclass
class Config:
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 오디오 샘플링 레이트
    TARGET_SR: int = 16000

    # 멜 프레임 길이(디코더 타겟 길이, frame 수)
    # - processor(audio_target=..., max_length=MAX_FRAMES)에서 사용
    # - 지나치게 길면 VRAM/시간 증가, 너무 짧으면 긴 발화가 잘려 품질 저하
    MAX_FRAMES: int = 2000

    # 배치 크기: SpeechT5 TTS는 메모리 부담이 크므로 작게 설정
    BATCH_SIZE: int = 3

    # 에포크 수: 단일 화자 작은 데이터셋에선 수백 epoch도 가능하지만 과적합 주의
    NUM_EPOCHS: int = 200

    # 학습률: AdamW의 기본적인 안정 영역
    LR: float = 1e-4

    # 오디오 최대 길이(초)
    # 너무 긴 샘플은 잘라서 메모리/학습 안정성 확보
    MAX_AUDIO_LEN: int = 10

    # weight decay: 과적합 완화(아주 작게)
    WEIGHT_DECAY: float = 1e-6

    # 텍스트 토큰 최대 길이(너무 긴 문장은 제거)
    # 자모 단위 토큰은 길이가 길어질 수 있으니 필터링 중요
    MAX_TOKENS: int = 200

    # train:test 비율
    TRAIN_RATIO: float = 0.98

    # 로컬 저장 경로
    MODEL_SAVE: Path = Path("./speecht5_kss_korean_jamo")

    # 자모 vocab 파일 경로
    JAMO_VOCAB: Path = Path("./jamo_vocab.txt")

    # Hugging Face Hub 설정
    HF_USER: str = "ahnhs2k"
    HF_REPO_NAME: str = "speecht5-korean-jamo"
    HF_PRIVATE: bool = False # 공개 리포로 설정
    HF_REPO_ID: str = ""

    # 랜덤 시드
    SEED: int = 42


# repo_id는 "유저/리포" 형태
Config.HF_REPO_ID = f"{Config.HF_USER}/{Config.HF_REPO_NAME}"

# 모델 저장 폴더가 없으면 생성(재실행해도 유지)
Config.MODEL_SAVE.mkdir(parents=True, exist_ok=True)


# Seed 고정(재현성)
def seed_all(seed: int):
    """
    재현성 확보를 위한 시드 고정.
    - random: 파이썬 기본 난수
    - numpy: 데이터 전처리/샘플링에서 사용될 수 있음
    - torch: 모델 초기화/드롭아웃/데이터 셔플 등
    - cuda: GPU 난수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_all(Config.SEED)
print("Device:", Config.DEVICE)

# Jamo 토크나이저 생성
# 여기서 하는 일:
# - jamo_vocab.txt를 읽어 "token -> id" 매핑(vocab)을 만듭니다.
# - HuggingFace tokenizers 라이브러리(빠른 토크나이저)를 이용해 WordLevel 토크나이저를 생성합니다.
# - 이 토크나이저를 PreTrainedTokenizerFast로 감싸 Transformers와 호환되게 만듭니다.
#
# WordLevel 사용 이유
# - 자모/placeholder를 이미 토큰 단위로 분해된 리스트로 넣어주기 때문에, 단순한 WordLevel 모델이 가장 적합합니다.
# - 즉, subword(BPE) 같은 분해가 필요 없고, 토큰=그 자체의 고정 매핑이 가장 안정적입니다.
if not Config.JAMO_VOCAB.exists():
    raise FileNotFoundError(
        f"[ERROR] jamo_vocab.txt를 찾을 수 없습니다: {Config.JAMO_VOCAB}\n"
        "먼저 jamo_vocab_builder.py를 실행해 vocab을 생성하세요."
    )

# vocab: dict[token:str -> id:int]
vocab = {}
with open(Config.JAMO_VOCAB, "r", encoding="utf-8") as f:
    for idx, tok in enumerate(f.read().splitlines()):
        vocab[tok] = idx

# WordLevel 모델은 (vocab, unk_token)만으로 고정 토큰 매핑을 제공
tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))

# pre_tokenizer=Whitespace:
# - 텍스트를 공백 기준으로 나누는 전처리
tok.pre_tokenizer = Whitespace()

# tokenizer.json 저장:
# - PreTrainedTokenizerFast는 tokenizer_file(.json)을 기반으로 로딩 가능
tokenizer_json_path = Config.MODEL_SAVE / "tokenizer.json"
tok.save(str(tokenizer_json_path))

# Transformers 호환 토크나이저
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_json_path),
    bos_token="<bos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
)

print("Tokenizer vocab size:", len(tokenizer))


# KSS 데이터셋 로드 + 리샘플
def load_and_prepare_kss():
    """
    Bingsu/KSS_Dataset 구조:
      - audio: {"array": np.ndarray, "sampling_rate": int}
      - original_script: 텍스트

    여기서 하는 전처리:
      1. decode=True로 audio를 실제 wave array로 로딩
      2. TARGET_SR(16k)로 리샘플링
      3. MAX_AUDIO_LEN 초보다 길면 자름(학습 안정성/메모리)
      4. "waveform", "sr" 컬럼을 추가하여 이후 파이프라인에서 사용하기 쉽게 함
    """
    # HF datasets로 로드
    kss = load_dataset("Bingsu/KSS_Dataset")

    # cast_column(Audio(...))는 audio 컬럼이 lazy decode될 수 있는 형태이기에
    # 실제 array로 decode하도록 설정
    kss = kss.cast_column("audio", Audio(decode=True))

    def safe_audio(batch):
        """
        datasets.map에서 각 샘플(또는 batch)을 받아 오디오를 정리합니다.

        주의:
        - 여기서는 batch라는 이름이지만 datasets.map에서 batched=False면
          샘플 1개 dict이 들어옵니다.
        """
        audio_info = batch["audio"]
        arr = audio_info["array"]
        sr = audio_info["sampling_rate"]

        # 리샘플: 서로 다른 sampling_rate가 섞여 있으면 모델 입력 분포가 흔들림
        if sr != Config.TARGET_SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=Config.TARGET_SR)
            sr = Config.TARGET_SR

        # 모델 입력은 float32 사용(메모리/속도)
        arr = arr.astype(np.float32)

        # 길이 제한: 너무 긴 샘플은 자르기
        max_len = int(Config.MAX_AUDIO_LEN * sr)
        if len(arr) > max_len:
            arr = arr[:max_len]

        batch["waveform"] = arr
        batch["sr"] = sr
        return batch

    # num_proc=4: 멀티프로세싱으로 전처리 속도 개선
    # 반환은 split dict 형태일 수 있으므로 ["train"] 접근
    kss_mod = kss.map(safe_audio, num_proc=4)["train"]
    return kss_mod


kss_mod = load_and_prepare_kss()


# 모델 / 프로세서 / Vocoder 로드
def load_models():
    """
    SpeechT5 TTS에 필요한 구성요소:
    - SpeechT5Processor: 텍스트/오디오 전처리 wrapper(토크나이저+feature extractor 역할)
    - SpeechT5ForTextToSpeech: TTS 모델 본체
    - SpeechT5HifiGan: 멜/특징을 waveform으로 바꾸는 vocoder

    여기서 중요한 커스터마이징:
    - processor.tokenizer를 "자모 토크나이저"로 교체
    - model.resize_token_embeddings(len(tokenizer))로 embedding 행렬 크기 변경
    - model.config.vocab_size도 함께 맞춰주기(일관성)
    """
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # 앞서 만든 자모 토크나이저로 교체(최중요!!!)
    # 이 처리를 안 하면 무슨 짓을 해도 SpeechT5는 한국어를 배울 수 없다.
    processor.tokenizer = tokenizer

    # 토큰 수가 바뀌었으므로 embedding 테이블 크기를 변경해야 한다.
    # 변경하지 않으면 input_ids가 embedding 범위를 벗어나 RuntimeError가 난다.
    # 이것도 처리하지 않으면 지옥이 펼쳐지므로 주의!!!
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)

    # 디바이스 이동
    model.to(Config.DEVICE)
    vocoder.to(Config.DEVICE)

    # guided attention loss 비활성
    # - guided attention loss는 텍스트-음성 정렬을 돕는 보조 손실인데,
    #   여기서는 사용하지 않도록 설정
    model.config.use_guided_attn_loss = False

    return processor, model, vocoder


processor, model, vocoder = load_models()


# Speaker Embedding (단일 화자)
def build_speaker_embedding(kss_mod):
    """
    단일 화자 데이터(KSS)이므로 speaker embedding을 하나만 고정 사용.

    파이프라인:
    1. WavLM speaker verification 모델 로드(WavLMForXVector)
    2. feature extractor(Wav2Vec2FeatureExtractor)로 입력 waveform을 텐서로 변환
    3. spk_model(**inputs)로 x-vector embedding 추출
    4. normalize하여 단위 벡터화(embedding scale 안정화)
    5. CPU로 가져와 저장/재사용 가능하게 함

    고정 이유
    - 멀티스피커 TTS는 샘플별 speaker embedding을 넣지만,
      단일 스피커 데이터는 동일 embedding이면 충분하다.
    - 이렇게 하면 데이터 전처리/콜레이터가 단순해지고 학습도 안정적이다.
    """
    spk_model = WavLMForXVector.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    ).to(Config.DEVICE)
    spk_model.eval()  # 드롭아웃/BN 등을 평가모드로 고정

    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    def get_spk_emb(waveform: np.ndarray) -> torch.Tensor:
        """
        waveform(np.ndarray) -> speaker embedding(torch.Tensor)

        return_tensors="pt": PyTorch 텐서로 반환
        padding=True: 길이가 다를 수 있으니 패딩 지원(여기서는 보통 1개 샘플)
        """
        inputs = fe(
            waveform,
            sampling_rate=Config.TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            emb = spk_model(**inputs)

            # emb.embeddings: (batch, emb_dim)
            # normalize: 코사인 기반 유사도에서 흔히 쓰는 정규화
            emb = torch.nn.functional.normalize(emb.embeddings, dim=-1)

        # squeeze().cpu(): (1,dim)->(dim)으로 만들고 CPU로 이동
        return emb.squeeze().cpu()

    # KSS 첫 샘플로 reference embedding 생성
    ref_spk_emb = get_spk_emb(kss_mod[0]["waveform"])
    return ref_spk_emb


ref_spk_emb = build_speaker_embedding(kss_mod)
print("Speaker embedding shape:", ref_spk_emb.shape)


# Text + Mel 전처리
def preprocess(batch):
    """
    샘플 1개에 대해 모델 학습에 필요한 input/label을 만든다.

    입력(원본 batch):
    - waveform: float32, 16kHz로 맞춰짐
    - original_script: 텍스트

    출력(dict):
    - input_ids: 자모+placeholder 토큰 ID 시퀀스
    - attention_mask: pad가 아닌 위치 = 1, pad = 0
    - labels: mel target (frames, n_mels) 형태

    설계
    - SpeechT5ForTextToSpeech는 forward에서 대략 다음을 기대한다:
      input_ids, attention_mask, labels(타겟), speaker_embeddings
    - labels는 decoder가 맞춰야 할 음향 타겟(teacher forcing)이다.
    """
    wf = batch["waveform"]

    # KSS 원문 스크립트. None 방어
    text_raw = batch.get("original_script", "")
    if text_raw is None:
        text_raw = ""

    # 훈련용 placeholder 전처리
    # - 숫자/단위/구두점을 <num>, <unit_kg>, <comma> 같은 의미 토큰으로 치환
    # - 훈련에서 normalize_korean(숫자->발음 전개)을 쓰지 않는 이유:
    #   데이터의 실제 발화와 전개 규칙이 불일치할 경우 정렬이 깨질 수 있음
    safe_text = inject_tokens_for_training(text_raw)

    # placeholder-aware 자모 분해
    # - "<...>" 토큰은 그대로 유지
    # - 한글 음절은 초/중/종성 자모로 분해하여 리스트로 반환
    jamo_seq = decompose_jamo_with_placeholders(safe_text)

    # 자모 토큰 리스트를 tokenizer에 전달하여 input_ids 생성
    # is_split_into_words=True:
    # - jamo_seq가 이미 "토큰 리스트"임을 의미
    # - tokenizer가 내부에서 다시 문자열을 쪼개지 않고 각 원소를 1 토큰으로 처리
    encoded = tokenizer(
        jamo_seq,
        is_split_into_words=True,
        add_special_tokens=True, # <bos>/<eos> 등 추가
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # mel target 생성
    # SpeechT5Processor의 audio_target 경로를 사용:
    # - wav -> mel/log-mel 형태의 타겟 특징을 생성
    # - padding="max_length"로 길이를 MAX_FRAMES에 맞춤(배치에서 stack 용이)
    audio_out = processor(
        audio_target=wf,
        sampling_rate=Config.TARGET_SR,
        truncation=True,
        max_length=Config.MAX_FRAMES,
        padding="max_length",
    )

    # 주의:
    # - processor(audio_target=...)의 결과에서 mel target은 "input_values"에 들어있다.
    # - shape: (batch=1, frames, n_mels)
    mel = audio_out["input_values"][0].astype(np.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": mel,
    }


# datasets.map으로 전체 전처리
# remove_columns=kss_mod.column_names:
# - 원본 컬럼(audio/original_script 등)을 제거하고 학습에 필요한 컬럼만 남김
dataset = kss_mod.map(preprocess, remove_columns=kss_mod.column_names)


def valid_text(x):
    """
    전처리 결과를 검사해 유효 샘플만 남긴다.

    체크:
    - input_ids 길이가 0이면 제거
    - 너무 긴 토큰 시퀀스는 제거(MAX_TOKENS)
      (자모 단위 토큰은 길이가 길어질 수 있어 필수)
    """
    ids = x["input_ids"]
    if isinstance(ids, int):
        length = 1
    else:
        length = len(ids)
    return 0 < length < Config.MAX_TOKENS


dataset = dataset.filter(valid_text)

# train/test 분할
split = int(len(dataset) * Config.TRAIN_RATIO)
tts = DatasetDict(
    {
        "train": dataset.select(range(split)),
        "test": dataset.select(range(split, len(dataset))),
    }
)

print("Train size:", len(tts["train"]), "| Test size:", len(tts["test"]))


# Data Collator
@dataclass
class DataCollator:
    """
    DataLoader가 배치를 만들 때 호출되는 콜레이터.

    역할
    - input_ids: 가변 길이이므로 padding 필요
    - attention_mask: pad가 아닌 곳을 1로 표시
    - labels(mel): 전처리에서 이미 max_length로 맞춰둬서 stack 가능
    - speaker_embeddings: 단일 화자 embedding을 batch 크기만큼 복제하여 제공

    collator에서 speaker embedding을 넣는 이유
    - forward 호출 시 매 배치마다 speaker_embeddings를 제공해야 함
    - 단일 화자라서 동일 embedding을 반복해서 쓰면 된다.
    """
    spk_emb: torch.Tensor  # [emb_dim]

    def __call__(self, features):
        # input_ids padding
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        input_ids = pad_sequence(
            input_ids,
            batch_first=True, # (B, T)
            padding_value=tokenizer.pad_token_id,
        )

        # attention_mask는 pad_token_id가 아닌 위치를 1로 만드는 방식으로 재계산
        # (전처리에서 만든 attention_mask를 쓰지 않고, 패딩 후 다시 계산하는 편이 안전)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # labels (mel) stack
        labels = torch.stack(
            [torch.tensor(f["labels"], dtype=torch.float32) for f in features]
        )

        # -100 마스킹:
        # - 일반적으로 손실 계산에서 무시할 위치를 -100으로 두는 관례가 많다.
        # - 여기서는 mel에서 0.0인 부분(패딩)을 -100으로 바꿔 loss에서 무시되게 유도
        # 주의:
        # - "실제 mel 값이 정확히 0.0인 프레임"이 있을 수도 있다는 이론적 리스크가 있다.
        # - 더 안전하게는 padding을 만들 때 별도 mask를 유지하거나, processor가 제공하는 attention/length 정보를 활용하는 방식이 좋다.
        labels = labels.masked_fill(labels.eq(0.0), -100.0)

        # speaker embedding 배치화
        B = len(features)
        spk = self.spk_emb.unsqueeze(0).repeat(B, 1)  # (B, emb_dim)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "speaker_embeddings": spk,
        }


collator = DataCollator(ref_spk_emb)

train_dl = DataLoader(
    tts["train"],
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)
val_dl = DataLoader(
    tts["test"],
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)


# Training Loop
# AdamW: 가중치 감쇠가 분리된 Adam 계열 옵티마이저(Transformer에서 표준)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=Config.LR,
    weight_decay=Config.WEIGHT_DECAY,
)

# AMP(Automatic Mixed Precision) scaler:
# - float16/ bfloat16 혼합 정밀도로 학습하면 속도/VRAM에 유리
# - scaler는 underflow를 막기 위해 loss scaling을 수행
scaler = torch.amp.GradScaler()


def train_one(epoch: int) -> float:
    """
    에포크 1회 학습.

    흐름:
    - model.train(): dropout 등 학습 모드
    - 배치마다:
      1. device로 이동
      2. optimizer.zero_grad()
      3. autocast 영역에서 forward
      4. scaler로 backward
      5. optimizer step + scaler update

    반환:
    - 평균 loss
    """
    model.train()
    total = 0.0

    for b in tqdm(train_dl, desc=f"Train {epoch}"):
        # 배치 dict의 모든 텐서를 device로 이동
        b = {k: v.to(Config.DEVICE) for k, v in b.items()}

        optimizer.zero_grad()

        # autocast:
        # - GPU에서 fp16 혼합 연산을 사용하도록 유도
        # - enabled=True는 강제 ON
        # - device_type은 "cuda" 또는 "cpu"
        with torch.amp.autocast(device_type=Config.DEVICE.type, enabled=True):
            out = model(**b) # out.loss가 존재

        scaler.scale(out.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += out.loss.item()

    return total / len(train_dl)


@torch.no_grad()
def eval_one(epoch: int) -> float:
    """
    검증 1회.
    - model.eval(): dropout 비활성
    - no_grad(): 그래프 저장 안 해서 메모리 절약
    """
    model.eval()
    total = 0.0

    for b in tqdm(val_dl, desc=f"Eval {epoch}"):
        b = {k: v.to(Config.DEVICE) for k, v in b.items()}
        out = model(**b)
        total += out.loss.item()

    return total / len(val_dl)


best = float("inf")
best_ckpt_path = Config.MODEL_SAVE / "speecht5-korean.pth"

if not args.skip_train:
    for e in range(1, Config.NUM_EPOCHS + 1):
        tr = train_one(e)
        val = eval_one(e)
        print(f"[{e}] train={tr:.4f} | val={val:.4f}")

        # best checkpoint 저장: val loss가 낮을수록 더 좋은 모델로 가정
        if val < best:
            best = val
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> New best saved to {best_ckpt_path}")
else:
    print("Training skipped (--skip_train enabled)")


# Best 모델 로드 + 로컬 아티팩트 저장
print(f"Loading best checkpoint from {best_ckpt_path}")
model.load_state_dict(torch.load(best_ckpt_path, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

# config에 speaker embedding path 기록
# - downstream 사용자가 hf_hub_download로 speaker_embedding.pth를 찾을 수 있도록 힌트 제공
model.config.speaker_embedding_path = "speaker_embedding.pth"

print("Saving artifacts to:", Config.MODEL_SAVE)

# 모델 저장(Transformers 표준)
# - config.json, pytorch_model.bin(또는 safetensors) 등이 생성
model.save_pretrained(Config.MODEL_SAVE)

# 토크나이저 저장(tokenizer.json + special_tokens_map 등)
tokenizer.save_pretrained(Config.MODEL_SAVE)

# vocoder 저장
# - vocoder는 별도 디렉터리에 저장해 충돌/관리 용이
vocoder_dir = Config.MODEL_SAVE / "vocoder"
vocoder.save_pretrained(vocoder_dir)

# 화자 임베딩 저장
spk_path = Config.MODEL_SAVE / "speaker_embedding.pth"
torch.save(ref_spk_emb, spk_path)

# korean_text_utils.py 복사 (Hub에서 import 가능하도록)
# - 이 파일이 모델 리포에 함께 올라가면, 추론 시 같은 전처리를 쉽게 재현 가능
utils_src = Path("korean_text_utils.py")
utils_dst = Config.MODEL_SAVE / "korean_text_utils.py"
if utils_src.exists():
    utils_dst.write_text(utils_src.read_text(encoding="utf-8"), encoding="utf-8")
    print("Copied korean_text_utils.py to model folder.")

print("All files saved.")

# demo_inference.py 생성 (prosody 기반)
def write_demo_script():
    """
    Hub에 올린 모델을 다운로드해서 바로 합성해볼 수 있는 데모 스크립트 생성.

    데모 설계:
    - prosody_split로 문장 단위 세그먼트 분할
    - segment마다 generate_speech 수행
    - segment 끝 구두점에 따라 prosody_pause로 무음 구간 삽입
    """
    demo_path = Config.MODEL_SAVE / "demo_inference.py"

    code = f'''
import numpy as np
import torch
import soundfile as sf

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download

from korean_text_utils import prosody_split, prosody_pause, decompose_jamo_with_placeholders

MODEL_ID = "{Config.HF_REPO_ID}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = {Config.TARGET_SR}


def load_model_and_assets():
    # Hub에서 fine-tuned 모델 로드
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to(DEVICE).eval()

    # 같은 repo의 tokenizer 로드(자모 토크나이저)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)

    # Vocoder는 base vocoder를 그대로 사용(또는 repo에 저장된 vocoder를 쓰는 방식도 가능)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE).eval()

    # speaker embedding 다운로드
    spk_path = hf_hub_download(repo_id=MODEL_ID, filename="speaker_embedding.pth")
    spk_emb = torch.load(spk_path)

    # (dim,)이면 (1,dim)으로 변환해 batch 차원 맞추기
    if spk_emb.dim() == 1:
        spk_emb = spk_emb.unsqueeze(0)
    spk_emb = spk_emb.to(DEVICE)

    return model, tokenizer, vocoder, spk_emb


def tts_with_prosody(text: str, out_path: str = "demo_inference_output.wav"):
    model, tokenizer, vocoder, spk_emb = load_model_and_assets()

    segments = prosody_split(text)
    audio_chunks = []

    for seg in segments:
        # seg → 자모 시퀀스(placeholder-aware)
        jamo_seq = decompose_jamo_with_placeholders(seg)

        enc = tokenizer(
            jamo_seq,
            is_split_into_words=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        enc = {{k: v.to(DEVICE) for k, v in enc.items()}}

        with torch.no_grad():
            wav = model.generate_speech(
                enc["input_ids"],
                speaker_embeddings=spk_emb,
                vocoder=vocoder,
            )
        audio_chunks.append(wav.cpu().numpy())

        # 구두점 기반 pause
        pause = prosody_pause(seg)
        if pause > 0:
            audio_chunks.append(np.zeros(int(SR * pause), dtype=np.float32))

    full = np.concatenate(audio_chunks, axis=0)
    sf.write(out_path, full, SR)
    print(f"Saved: {{out_path}}")


if __name__ == "__main__":
    text = "안녕하세요. 오늘은 3.5km를 걸었습니다. 숫자와 구두점 처리 데모입니다!"
    tts_with_prosody(text)
'''
    demo_path.write_text(code, encoding="utf-8")
    print("demo_inference.py written to", demo_path)


write_demo_script()


# Hugging Face Hub 업로드
def create_and_push_to_hub():
    """
    HF Hub 업로드:
    - create_repo: 리포가 없으면 생성, 있으면 OK
    - upload_folder: MODEL_SAVE 폴더 안의 파일들을 전부 업로드

    포함되는 것:
    - config.json / model weights / tokenizer.json / special token maps
    - vocoder 디렉터리
    - speaker_embedding.pth
    - korean_text_utils.py
    - README.md
    - demo_inference.py
    """
    api = HfApi()
    api.create_repo(
        repo_id=Config.HF_REPO_ID,
        private=Config.HF_PRIVATE,
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=Config.HF_REPO_ID,
        folder_path=str(Config.MODEL_SAVE),
        path_in_repo="",
        commit_message="Upload SpeechT5 Korean TTS artifacts",
    )

    print("Pushed to Hugging Face Hub:", Config.HF_REPO_ID)


create_and_push_to_hub()

print("All done.")
