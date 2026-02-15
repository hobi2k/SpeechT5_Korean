"""
한국어 TTS용 텍스트 처리 모듈 (훈련/추론 모두 지원)

이 모듈이 하는 일
- "훈련용 전처리"와 "추론용 전처리"를 분리합니다.
  - 훈련(training): alignment(정렬) 안전성을 최우선으로 하여
    숫자/단위/구두점을 의미 토큰 placeholder로 바꾼 뒤 자모 분해.
  - 추론(inference): 사람이 읽는 자연스러운 발음 형태로 숫자/단위를 풀어쓴 뒤
    문장부호 기준으로 segment를 나누고, segment별 pause 길이 등을 결정.

분리 이유
- 훈련 데이터는 대개 음성-텍스트 페어로 이미 녹음된 발화에 대응됩니다.
  여기서 텍스트를 과하게 변형(예: 숫자 1203 -> 천이백삼)하면,
  원래 녹음에서 실제로 말한 방식(천이백삼인지 일이공삼인지 등)과 불일치하여
  학습이 흔들릴 수 있습니다. 이는 alignment mismatch입니다.
- 따라서 훈련에서는 "읽기 변환(자연 발음화)"을 하지 않고,
  대신 <num>, <unit_kg>, <comma>처럼 의미 토큰으로만 치환해
  텍스트 길이/구조 변형을 최소화합니다.
"""

import re
import unicodedata

# 1. 숫자 -> 한국어 읽기(추론용)

# 한 자리 숫자를 한국어 한자어(영/일/이/삼...)로 바꾸기 위한 매핑
_NUMBER_KOR = {
    0: "영",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구"
}

# 10^i 단위(천/백/십/일)를 표현하기 위한 작은 단위 리스트
# i=0: "" (일의 자리)
# i=1: "십"
# i=2: "백"
# i=3: "천"
_UNIT_KOR = ["", "십", "백", "천"]

# 10,000 단위로 끊을 때 붙는 큰 단위(만/억/조/경...)
# idx=0: "" (0~9999 블록)
# idx=1: "만" (10^4)
# idx=2: "억" (10^8)
# idx=3: "조" (10^12)
# idx=4: "경" (10^16)
_BIG_UNIT = ["", "만", "억", "조", "경"]


def number_to_korean(num_str: str) -> str:
    """
    숫자 문자열을 '발화 친화적인 한국어 읽기'로 바꿉니다. (추론용)

    예:
    - "1203" -> "천이백삼"
    - "3.5" -> "삼 점 오"

    설계 포인트:
    - 입력을 문자열(str)로 받는 이유:
      1. 원문에서의 형태(선행 0, 매우 큰 수 등)를 보존할 여지가 있음
      2. 정규식 매칭(re.sub)에서 그대로 넘겨받기 편함
    - 여기 구현은 한자어 수사(일/이/삼...) 기반의 단순 버전입니다.
      (순우리말 "한/두/세" 같은 형태는 다루지 않습니다.)
    """
    # 소수 처리:
    # "3.5" -> 좌측 정수부는 기존 로직으로 읽고,
    # 우측 소수부는 각 자리수를 "삼 점 오"처럼 한 자리씩 읽습니다.
    if "." in num_str:
        a, b = num_str.split(".")
        left = number_to_korean(a)

        # 소수부 b는 "5" "14" 등 문자열이므로
        # 각 문자 x를 int로 바꾼 뒤 _NUMBER_KOR로 한글 숫자를 얻습니다.
        # "314" -> "삼 일 사"처럼 자리별로 띄어 읽게 됩니다.
        right = " ".join(_NUMBER_KOR[int(x)] for x in b)
        return f"{left} 점 {right}"

    # 정수부 처리
    num = int(num_str)
    if num == 0:
        return "영"

    result = [] # 10,000 단위 블록별 변환 결과를 담습니다.
    idx = 0 # _BIG_UNIT 인덱스(0: '', 1:'만', 2:'억' ...)

    # 핵심 아이디어:
    # - 한국어 큰 수 읽기는 10,000(만) 단위로 끊어서 읽는 구조가 자연스럽습니다.
    # - 따라서 num을 10000으로 나누며 "아래 4자리 블록"을 반복 추출합니다.
    #
    # divmod(num, 10000) -> (몫, 나머지)
    # - num: 12345678
    # -> num=1234, digit=5678 (첫 블록)
    # -> num=0, digit=1234 (다음 블록)
    while num > 0:
        num, digit = divmod(num, 10000)

        # digit은 0~9999 범위. 0이면 해당 블록은 스킵합니다.
        if digit > 0:
            part = _convert_under_10000(digit) # '천이백삼십사' 같은 형태
            result.append(part + _BIG_UNIT[idx]) # 블록에 '만/억/조...'를 붙임

        idx += 1  # 다음 10,000 단위로 이동

    # result는 하위 블록부터 append 되었으므로 reverse해서 붙입니다.
    # 또한 중간에 공백을 넣어 '억 만' 사이 구분이 보이도록 합니다.
    return " ".join(reversed(result)).strip()


def _convert_under_10000(n: int) -> str:
    """
    0 <= n <= 9999 범위의 숫자를 한국어로 읽는 부분 변환기.

    예:
    - 1203 -> "천이백삼"
    - 9000 -> "구천"
    - 1010 -> "천십"
    """
    res = []

    # i는 자리수 인덱스: 0(일) 1(십) 2(백) 3(천)
    # digit = (n // (10 ** i)) % 10  -> 해당 자리 숫자
    for i, unit in enumerate(_UNIT_KOR):
        digit = (n // (10 ** i)) % 10
        if digit != 0:
            # 예: digit=2, unit="백" -> "이백"
            res.append(_NUMBER_KOR[digit] + unit)

    # res는 아래자리부터 쌓이므로 reversed로 뒤집어서 정방향으로 결합합니다.
    return "".join(reversed(res))

# 추론용 텍스트 정규화(normalize)
def normalize_korean(text: str) -> str:
    """
    추론용 텍스트 정규화 함수.

    하는 일:
    - 단위(km, kg, %, ml 등)를 한국어 발음어로 치환
    - 숫자(정수/소수)를 한국어 읽기로 변환(number_to_korean)
    - 공백을 정리

    경고:
    - 훈련에서는 절대 사용 금지.
      이유:
      - 원래 데이터의 텍스트-음성 정렬을 깨뜨릴 수 있습니다.
      - 예: 녹음에서 "이십사"라고 안 하고 "이사"처럼 축약/구어체일 수도 있고,
        숫자 읽기 규칙이 데이터와 불일치하면 모델이 학습 신호를 혼동합니다.
    """

    # unit_map:
    # 원문에서 흔히 등장하는 단위 문자열을 사람이 말하는 형태로 바꾸기 위한 맵입니다.
    # 주의: 단순 replace이므로 "kg"가 다른 단어의 일부로 포함된 경우도 치환될 수 있습니다.
    # 해당 함수 실제 사용 시에서는 정규식 경계 처리(예: 숫자 뒤에만 오는 kg 등)를 추가합니다.
    unit_map = {
        "km": " 킬로미터",
        "kg": " 킬로그램",
        "cm": " 센티미터",
        "mm": " 밀리미터",
        "%": " 퍼센트",
        "ml": " 밀리리터",
    }

    # 단위 치환
    for k, v in unit_map.items():
        text = text.replace(k, v)

    # 숫자 치환: 정규식으로 숫자 패턴을 찾고, 매치된 문자열을 한국어 읽기로 바꿉니다.
    # 패턴: \d+(\.\d+)?
    # - \d+        : 1개 이상의 숫자
    # - (\.\d+)?   : (소수점 + 숫자들) 이 있을 수도, 없을 수도
    def repl(m):
        # m.group()은 매치된 원문 숫자 문자열(예: "1203", "3.14")입니다.
        return number_to_korean(m.group())

    text = re.sub(r"\d+(\.\d+)?", repl, text)

    # 공백 정리:
    # - 여러 개의 공백/탭/개행을 하나의 스페이스로 줄이고
    # - 양끝 공백을 제거합니다.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# prosody segmentation (추론 시)
def prosody_split(text: str):
    """
    추론 시, “문장부호 기반”으로 텍스트를 segment 단위로 쪼갭니다.

    - 긴 문장을 한 번에 합성하면:
      1. 모델 입력 길이가 길어져 속도가 느려지고
      2. 메모리 부담이 커지고
      3. 운율/호흡이 어색해질 수 있습니다.
    - 따라서 문장 단위(혹은 ! ? . 등)로 나누어 합성한 뒤
      segment 사이에 pause를 주는 전략을 사용합니다.

    구현 흐름:
    1) normalize_korean으로 숫자/단위를 자연 발화형으로 변환
    2) [.!?] 를 기준으로 split하되, 구분자 자체도 보존
       - re.split(r"([.!?])", ...)에서 괄호()는 “캡처 그룹”이라
         split 결과에 구분자도 리스트에 포함됩니다.
    3) (본문 + 구두점) 단위로 재결합하여 seg 리스트를 만듭니다.
    """
    text = normalize_korean(text)

    # 예: "안녕! 오늘 3.5kg."
    # re.split 결과(대략):
    # ["안녕", "!", " 오늘 삼 점 오 킬로그램", ".", ""]
    parts = re.split(r"([.!?])", text)

    segs = []
    # i=0,2,4... 본문 / i+1은 구두점이 되도록 2칸씩 점프
    # len(parts)-1 까지만 도는 이유: i+1 접근 안전성 확보
    for i in range(0, len(parts) - 1, 2):
        segs.append((parts[i] + parts[i + 1]).strip())
    return segs


def prosody_pause(segment: str) -> float:
    """
    segment 끝의 구두점에 따라 “세그먼트 뒤에 넣을 pause 길이(초)”를 반환합니다.

    사용 예(추론 파이프라인):
    - segments = prosody_split(text)
    - for seg in segments:
        wav = tts(seg)
        out.append(wav)
        out.append(silence(duration=prosody_pause(seg)))

    왜 heuristic(규칙 기반)인가?
    - 모델이 pause까지 완벽히 생성하도록 학습하는 것도 가능하지만,
      간단한 데모/프로덕션 파이프라인에서는
      “문장부호 > pause 길이” 규칙이 비용 대비 효과가 좋습니다.
    """
    if segment.endswith("."):
        return 0.35
    if segment.endswith(","):
        return 0.22
    if segment.endswith("!") or segment.endswith("?"):
        return 0.45
    return 0.18


# 훈련용 placeholder 전처리(alignment-safe)

# 훈련 시에는 자연 발화형 풀어쓰기를 하지 않고,
# 숫자/단위/기호를 “의미 토큰”으로 치환하여
# 텍스트 구조 변화와 alignment 리스크를 줄입니다.

# 단위 placeholder
TRAINING_UNITS = {
    "km": "<unit_km>",
    "kg": "<unit_kg>",
    "cm": "<unit_cm>",
    "mm": "<unit_mm>",
    "ml": "<unit_ml>",
    "%": "<unit_percent>",
}

# 구두점 placeholder
TRAINING_PUNCT = {
    ",": "<comma>",
    ".": "<period>",
    "?": "<question>",
    "!": "<exclamation>",
    ":": "<colon>",
    ";": "<semicolon>",
    "-": "<dash>",
    "\"": "<quote>",
    "“": "<quote>",
    "”": "<quote>",
    "(": "<lparen>",
    ")": "<rparen>",
}


def inject_tokens_for_training(text: str) -> str:
    """
    훈련 데이터 텍스트를 “alignment-safe” 형태로 바꿉니다.

    정책:
    - normalize_korean() 절대 사용 X (발음 풀어쓰기는 alignment를 깨기 쉬움)
    - 숫자는 <num> (정수/소수 모두)
    - 단위는 <unit_x>
    - 구두점은 <comma> 등 placeholder 토큰으로 치환

    왜 placeholder인가?
    - 모델이 "숫자/단위/구두점"의 존재를 학습하되,
      실제 발음 형태는 데이터/모델/후처리에 맡기는 전략 채텍
    - 특히 숫자 읽기는 케이스가 너무 다양해(연도, 전화번호, 소수, 범위, 약어…)
      훈련 데이터가 충분하지 않으면 규칙 기반 처리로 분리
    """

    # 단위 치환을 우선 처리
    for k, v in TRAINING_UNITS.items():
        text = text.replace(k, v)

    # 숫자 -> <num>
    # 소수점 포함까지 커버하는 동일 정규식을 사용
    # 이렇게 하면 "3.5"도 통째로 <num> 하나로 바뀝니다.
    # (좀 더 정밀하게 하려면 "<num> <period> <num>" 같은 분해 정책도 가능)
    text = re.sub(r"\d+(\.\d+)?", "<num>", text)

    # 구두점 -> placeholder
    # 순수 replace이므로, 따옴표 종류(“ ”)까지 모두 동일 토큰으로 통일합니다.
    # 이 단계는 토크나이저가 실제 문자를 배울 필요 없이 의미 토큰만 보면 되게 해줍니다.
    for src, tok in TRAINING_PUNCT.items():
        text = text.replace(src, tok)

    # 공백 정리: 다양한 공백 문자들을 전부 하나의 공백으로 변환합니다.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 자모 분해 (placeholder-aware)
def decompose_jamo_with_placeholders(text: str):
    """
    텍스트를 토큰 리스트로 변환합니다.

    핵심 목표:
    - "<...>" 형태의 placeholder 토큰은 절대 분해하지 않고 그대로 유지
    - 나머지 한글 음절(가-힣)은 초/중/종성 자모로 분해
    - 한글 음절이 아닌 문자는 그대로 유지(예: 공백, 영문, 기타 기호)

    반환:
    - tokens: List[str]
      예: "안녕<comma>3kg" (훈련 전처리 후라면)
          -> ["ᄋ","ᅡ","ᆫ","ᄂ","ᅧ","ᆼ","<comma>","<num>","<unit_kg>"]
    """

    # placeholder 정규식:
    # - < 로 시작해서 > 로 끝나는 가장 단순한 토큰 형태를 가정합니다.
    # - "<unit_kg>", "<comma>" 같은 형태가 여기에 해당합니다.
    pattern = r"<[^>]+>"

    tokens = []  # 최종 토큰 리스트
    pos = 0      # 현재까지 처리한 문자열 위치

    # re.finditer로 placeholder 토큰들을 순서대로 찾습니다.
    # 각 match m에 대해:
    # - m.span() = (start, end)
    # - start~end 구간이 "<...>" 토큰
    for m in re.finditer(pattern, text):
        start, end = m.span()

        # placeholder 앞쪽의 일반 텍스트 구간
        normal = text[pos:start]

        # normal 구간을 문자 단위로 순회하며
        # 한글 음절은 자모로 분해, 그 외는 그대로 토큰으로 추가
        for ch in normal:
            # unicodedata.name(ch, ""):
            # - 유니코드 문자 이름을 반환합니다.
            # - 한글 음절(가-힣)은 이름에 "HANGUL SYLLABLE"이 포함됩니다.
            if "HANGUL SYLLABLE" in unicodedata.name(ch, ""):
                tokens.extend(_decompose_jamo_char(ch))
            else:
                tokens.append(ch)

        # placeholder 자체는 하나의 토큰으로 추가
        tokens.append(text[start:end])

        # pos를 placeholder 끝으로 옮겨 다음 구간을 처리
        pos = end

    # 마지막 placeholder 이후의 tail(꼬리) 구간 처리
    tail = text[pos:]
    for ch in tail:
        if "HANGUL SYLLABLE" in unicodedata.name(ch, ""):
            tokens.extend(_decompose_jamo_char(ch))
        else:
            tokens.append(ch)

    return tokens


def _decompose_jamo_char(ch):
    """
    한글 음절(가-힣) 하나를 초/중/종성 자모로 분해합니다.

    유니코드 한글 음절 조합 규칙(핵심):
    - '가' (U+AC00)부터 음절이 순서대로 배치되어 있습니다.
    - 각 음절은 초성 19개 * 중성 21개 * 종성 28개(받침없음 포함)
      조합으로 인덱싱됩니다.
    - 공식:
        code = ord(ch) - 0xAC00
        choseong_index = code // 588
        jungseong_index = (code % 588) // 28
        jongseong_index = code % 28
      (왜 588인가? 21*28=588)

    반환:
    - out: List[str]
      예: '강' -> ['ᄀ','ᅡ','ᆼ']
      받침이 없으면 종성 자모는 생략합니다.
    """
    code = ord(ch) - 0xAC00

    # 초성: 0x1100(ᄀ) + choseong_index
    choseong = chr(0x1100 + (code // 588))

    # 중성: 0x1161(ᅡ) + jungseong_index
    jungseong = chr(0x1161 + ((code % 588) // 28))

    # 종성 인덱스(0이면 받침 없음)
    jong = code % 28

    out = [choseong, jungseong]

    # 종성 처리:
    # - jong=0: 받침 없음 -> 추가하지 않음
    # - jong>0: 받침 있음
    #   종성 자모는 0x11A8부터 시작하지만,
    #   여기서는 "0x11A7 + jong"로 계산합니다.
    #   (jong=1 -> 0x11A8, jong=27 -> 0x11C2)
    if jong > 0:
        out.append(chr(0x11A7 + jong))

    return out


"""
추후 개선 포인트

1) normalize_korean의 단위 치환 안정성
- 현재는 text.replace("kg", " 킬로그램")처럼 단순 치환입니다.
- 더 안전하게 하려면:
  - 숫자 뒤에 오는 kg만 치환: r"(\d)\s*kg\b" 같은 패턴
  - "program" 안의 "g" 같은 오치환 방지
  등을 적용합니다.

2) number_to_korean의 읽기 규칙 고도화
- '일십'을 '십'으로 줄이는 규칙(10~19 등)
- '일백'->'백', '일천'->'천'
- '1,234' 콤마 포함 숫자, 전화번호/연도 읽기
- 음성 데이터의 스타일(한자어/고유어)에 따른 분기
가 필요할 수 있습니다.

3) 훈련/추론 정책 분리의 일관성
- 훈련에서 <comma>, <period>, <short_pause> 등을 넣었다면
  추론에서도 동일한 토큰 정책을 쓰거나,
  추론은 휴리스틱 pause(현재 prosody_pause)로 처리하는 등
  “모델이 본 입력 분포”와 “추론 입력 분포”가 너무 벌어지지 않게 설계 수정.

4) placeholder 패턴 확장
- 현재는 "<...>"만 잡습니다.
- 만약 토큰 안에 '>'가 들어갈 가능성이 있거나,
  nested 토큰이 있다면 파서가 더 필요합니다.

5) G2P 모듈과의 연동
- 한국어 발음 변환(G2P) 모듈과 연동하여
  숫자/단위 읽기, 발음 변환 등을 통합 관리하는 방안 고려
"""
