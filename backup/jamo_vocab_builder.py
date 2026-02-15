"""
jamo_vocab_builder.py
한국어 TTS용 자모 기반 vocab builder

이 파일의 목적
- "자모(초/중/종성) + 기호/숫자/단위/운율 토큰"으로 구성된 vocab(어휘 목록)을 만들어
  jamo_vocab.txt로 저장합니다.
- 이 vocab은 텍스트를 토큰 ID로 바꾸는 토크나이저(Tokenizer)가 참조하는 기준표가 됩니다.

TTS에서의 vocab
- 텍스트 입력을 토큰 시퀀스로 받는 TTS는, 입력 문장을 먼저 토큰화하여
  정수 ID로 변환하고, 모델은 그 ID 시퀀스를 바탕으로 음성 특징(멜 스펙트로그램 등)을 생성합니다.
- vocab은 "어떤 단위를 토큰으로 삼을지"를 정의합니다.
  - 완성형(가-힣) 기반: 토큰 수가 커지고(모든 글자를 포함하려면), 미등록(OOV) 처리 난감.
  - 자모 기반: 초/중/종성 조합으로 모든 한글을 표현 가능 -> vocab이 작고 일반화가 쉬움.
- 또한 “쉼표, 물음표, 단위(km, kg), pause(짧은 쉼)” 같은 것을 별도 토큰으로 두면,
  모델이 운율/억양/리듬을 더 안정적으로 학습할 수 있습니다.

uv run sp5_kor/text/jamo_vocab_builder.py
"""

from pathlib import Path

# SPECIAL: 모델/토크나이저가 거의 항상 필요로 하는 특수 토큰들
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]

# <pad> : padding 토큰
# - 배치 학습에서는 문장 길이가 제각각이므로 가장 긴 문장 길이에 맞춰 짧은 문장을 채워야 합니다.
# - 그 "빈 칸"을 <pad>로 채우고 attention_mask 등으로 무시합니다.
#
# <unk> : unknown 토큰
# - vocab에 없는 문자를 만났을 때 대체로 들어가는 토큰입니다.
# - 자모 vocab을 쓰면 한글 OOV는 거의 줄지만, 예외 문자(특수기호/외국문자 등)가 남습니다.
#
# <bos> : beginning-of-sequence 토큰
# - 문장의 시작을 명시합니다. (모델/학습 설정에 따라 쓰지 않을 수도 있지만 있어두면 유용)
#
# <eos> : end-of-sequence 토큰
# - 문장의 끝을 명시합니다. 생성 모델에서 “여기서 끝”을 학습시키는 신호가 됩니다.

# 한글 자모 코드포인트 범위로 초/중/종성을 만들기
# 파이썬에서 chr(코드포인트)는 해당 유니코드 문자를 만들어 줍니다.
# 여기서는 “한글 자모” 블록의 유니코드 코드포인트 범위를 이용합니다.
#
# 주의:
# - range(a, b)는 a 이상 b 미만입니다.
# - 즉, 끝값을 포함하려면 b를 (마지막코드+1)로 줘야 합니다.

# 초성(Choseong): U+1100 ~ U+1112 (19개)
CHOSEONG = [chr(c) for c in range(0x1100, 0x1113)]
# - 0x1100(ㄱ)부터 0x1112(ㅎ)까지의 "한글 자모(초성)"" 문자들을 리스트로 생성합니다.
# - 결과 예: ['ᄀ','ᄁ','ᄂ', ... , 'ᄒ']

# 중성(Jungseong): U+1161 ~ U+1175 (21개)
JUNGSEONG = [chr(c) for c in range(0x1161, 0x1176)]
# - 0x1161(ㅏ)부터 0x1175(ㅣ)까지의 "한글 자모(중성)"을 생성합니다.

# 종성(Jongseong): U+11A8 ~ U+11C2 (27개)
JONGSEONG = [chr(c) for c in range(0x11A8, 0x11C3)]
# - 종성은 받침 자모들로 구성됩니다.
# - 여기서는 "받침 없음"은 별도 처리를 한다는 전제를 깔고(혹은 토크나이저가 처리),
#   받침이 있는 경우의 자모만 vocab에 넣는 방식입니다.

# 기호(PUNCT): 문장부호를 '문자 자체'가 아니라 '의미 토큰'으로 넣는 설계
PUNCT = [
    "<comma>", "<period>", "<question>", "<exclamation>",
    "<colon>", "<semicolon>", "<dash>", "<quote>",
    "<lparen>", "<rparen>"
]

# 콤마(,)와 마침표(.)가 실제 텍스트에 다양한 형태로 등장할 수 있습니다.
# - 예: "," "，" ".", "…" 등
# 토크나이저에서 이것들을 정규화(normalize)해서 모두 <comma>, <period> 같은 의미 토큰으로 바꿉니다.
# TTS에서는 문장부호가 “쉼/억양”에 직접 영향을 주므로, 이를 명확히 표현합니다.

# 숫자(NUM): 숫자를 그대로 읽지 않고 별도 처리할 때 쓰는 토큰
NUM = ["<num>"]

# 숫자를 "그대로 문자 토큰화"하면 vocab에 '0'~'9'를 넣고 처리할 수도 있지만, TTS에서는 숫자를 한국어로 읽는 규칙(예: 12 -> "십이", 3.14 -> "삼점일사")이 중요합니다.
# 따라서 실제 파이프라인에서는:
# (a) 전처리 단계에서 숫자를 한국어 발음으로 풀어쓰기
# (b) 혹은 숫자 구간을 <num>으로 치환하고 별도 숫자 처리 모듈을 둠
# 여기서는 (b)도 가능하도록 토큰을 준비한 형태입니다.

# 단위: 숫자+단위를 다룰 때의 의미 토큰
UNITS = [
    "<unit_km>", "<unit_kg>", "<unit_cm>", "<unit_mm>",
    "<unit_ml>", "<unit_percent>"
]

# 예시:
# - "3kg"  -> "<num> <unit_kg>" 또는 "삼 킬로그램" 식으로 전처리
# - "%" 같은 기호도 정규화해서 <unit_percent>로 통일하면 모델이 다양한 표기를 하나의 패턴으로 학습할 수 있습니다.

# 운율/쉼표(PROSOY): pause 토큰을 명시적으로 넣는 설계
PROSODY = ["<short_pause>", "<medium_pause>", "<long_pause>"]

# TTS에서 "쉼"은 매우 중요합니다.
# - 문장부호는 쉼을 암시하지만, 실제 발화에서는 쉼 길이가 다양합니다.
# - 학습 데이터에서 쉼 길이에 대한 정답 신호가 없다면 모델이 임의로 추정해야 하는데, pause 토큰을 넣어주면 텍스트에서 운율 힌트를 더 강하게 줄 수 있습니다.
#
# - pause 토큰이 실제로 의미 있으려면, 전처리에서 언제 어떤 pause를 넣을지 정책이 있어야 합니다.
#   (예: 쉼표=short_pause, 문장끝=long_pause 등)

# 최종 VOCAB 구성: 순서가 곧 ID입니다.
VOCAB = SPECIAL + CHOSEONG + JUNGSEONG + JONGSEONG + PUNCT + NUM + UNITS + PROSODY

# - 많은 WordLevel/단순 vocab 기반 토크나이저는 "파일에 적힌 줄 순서"대로 토큰 ID를 부여합니다.
#   예: 0번=<pad>, 1번=<unk>, 2번=<bos> ...
# - 따라서 vocab의 "순서"는 재현성(reproducibility)에 직결됩니다.
# - 기존에 학습한 모델이 있다면, vocab 순서를 절대로 바꾸면 안 됩니다.
#   (바꾸면 같은 텍스트가 다른 ID 시퀀스로 변해 모델이 망가집니다.)

# vocab 저장 함수
def save_vocab(path: Path | None = None):
    """
    VOCAB 리스트를 텍스트 파일로 저장합니다.

    Args:
        path (Path | str):
            저장할 파일 경로.
            기본값은 현재 작업 디렉토리의 "jamo_vocab.txt".

    동작:
        - 파일을 UTF-8로 열고
        - VOCAB의 각 토큰을 한 줄씩 기록합니다.
        - 저장 완료 후 토큰 개수도 함께 출력합니다.

    출력 파일 포맷(중요):
        - 1줄 = 1토큰
        - 예:
            <pad>
            <unk>
            <bos>
            <eos>
            ᄀ
            ᄁ
            ...
    """
    # Path 객체든 문자열이든 open이 가능하지만,
    # 여기서는 명시적으로 Path를 기본값으로 사용했습니다.
    if path is None:
        path = Path(__file__).resolve().parent / "jamo_vocab.txt"

    with open(path, "w", encoding="utf-8") as f:
        for t in VOCAB:
            f.write(t + "\n")

    # len(VOCAB): 전체 토큰 수
    # 나중에 토크나이저/모델 config의 vocab_size와 일치하는지 점검할 때 사용할 수 있습니다.
    print(f"Saved vocab: {path} | total={len(VOCAB)}")


# 스크립트로 직접 실행될 때 save_vocab 호출
if __name__ == "__main__":
    save_vocab()

"""
추후 업데이트를 위한 추가 노트

1. 받침 없음 처리
- 종성(JONGSEONG) 리스트에는 “받침 없음”이 들어있지 않습니다.
- 토크나이저에서:
  - 받침이 없으면 종성 토큰을 생략하거나,
  - <no_jong> 같은 명시 토큰을 vocab에 추가하는 방식 중 하나를 택합니다.
- SpeechT5 같은 모델에선 “입력 길이”가 운율/길이 예측에 영향을 주기도 하므로,
  어떤 방식이 더 안정적인지는 실험적으로 결정해야 합니다.

2. PUNCT, PROSODY의 실제 효용은 전처리 정책에 달렸습니다.
- vocab에 넣는 것만으로는 의미가 없습니다.
- 데이터 전처리에서 실제 텍스트를:
  - "," -> <comma>
  - "." -> <period>
  - 문장 경계/쉼표 규칙에 따라 <short_pause> 삽입
  같은 방식으로 변환해줘야 모델이 학습합니다.

3. vocab 변경 시 주의
- 이미 학습한 모델이 있으면 vocab은 “불변”으로 취급하는 게 원칙입니다.
- 토큰을 추가하고 싶다면:
  - 새 모델을 처음부터 다시 학습하거나,
  - (일부 프레임워크에서는) 추가 토큰을 뒤에 붙여 vocab_size를 늘리고
    embedding을 리사이즈하는 전략을 씁니다.
  다만 TTS는 텍스트 임베딩 변화에 민감할 수 있어 신중해야 합니다.
"""
