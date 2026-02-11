from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def build_tokenizer(vocab_path: Path, output_dir: Path) -> PreTrainedTokenizerFast:
    """
    자모 vocab 파일로 Hugging Face fast tokenizer를 생성한다.

    동작:
    1. `vocab_path`의 토큰 라인을 읽어 token->id 사전을 만든다.
    2. `WordLevel` 토크나이저를 생성한다.
    3. 토크나이저를 `tokenizer.json`으로 저장한다.
    4. `PreTrainedTokenizerFast` 래퍼를 반환한다.

    Args:
        vocab_path: 1줄 1토큰 형식의 vocab 파일 경로.
        output_dir: 생성된 `tokenizer.json`과 tokenizer 메타 파일 저장 디렉터리.

    Returns:
        PreTrainedTokenizerFast: 학습/추론에서 바로 사용할 수 있는 토크나이저 객체.

    Raises:
        FileNotFoundError: vocab 파일이 존재하지 않는 경우.
    """
    # vocab 파일 존재 여부 확인.
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"jamo vocab not found: {vocab_path}. Run sp5_kor/text/jamo_vocab_builder.py first."
        )

    # 줄 순서를 그대로 토큰 ID로 사용한다.
    vocab: dict[str, int] = {}
    for idx, tok in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
        vocab[tok] = idx

    # 공백 단위 토큰 분리를 사용하는 WordLevel tokenizer 구성.
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = Whitespace()

    # tokenizer.json 저장.
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = output_dir / "tokenizer.json"
    tk.save(str(tokenizer_json))

    # transformers 호환 fast tokenizer 반환.
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
