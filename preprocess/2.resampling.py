# 오디오 처리 및 리샘플링 라이브러리
import librosa
# 오디오 파일 읽고 쓰는 라이브러리
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# 원본 wav 파일들이 있는 디렉토리 경로 (원래 샘플율 48kHz)
SRC_DIR = Path("/mnt/d/tts_data/yae_ko/audio")
# 리샘플링된 wav를 저장할 출력 디렉토리 경로 (목표 샘플율 16.00kHz)
# data.py에서 리샘플링을 수행하긴 하지만, 미리 수행한다.
DST_DIR = Path("/mnt/d/tts_data/yae_ko/sp5")

# 목표 샘플링 레이트를 1600Hz로 설정
TARGET_SR = 16000

# 출력 디렉토리 생성 (부모 폴더도 함께 생성, 이미 있으면 무시)
DST_DIR.mkdir(parents=True, exist_ok=True)

# SRC_DIR에서 모든 .wav 파일을 찾아 리스트로 변환
wav_files = list(SRC_DIR.glob("*.wav"))
# 발견된 wav 파일 개수를 출력 (f-string으로 동적 텍스트 삽입)
print(f"Found {len(wav_files)} wav files")

for wav_path in tqdm(wav_files):
    try:
        # 로드 (원 SR 자동 인식)
        # librosa.load로 오디오 파일 로드 및 리샘플링 시작
        audio, sr = librosa.load(
            # 로드할 오디오 파일 경로
            wav_path,
            # sr 파라미터: 목표 샘플링 레이트로 자동 리샘플링
            sr=TARGET_SR,
            # mono=True: 스테레오를 모노로 변환 (채널 1개)
            mono=True
        )

        # 출력 파일 경로 생성 (원본 파일명 유지)
        out_path = DST_DIR / wav_path.name

        # 저장 (16-bit PCM)
        # soundfile.write로 오디오 파일 저장 시작
        sf.write(
            # 저장할 파일 경로
            out_path,
            # 저장할 오디오 데이터
            audio,
            # 샘플링 레이트
            TARGET_SR,
            # subtype: 16비트 정수형 PCM 포맷으로 저장
            subtype="PCM_16"
        )

    except Exception as e:
        print(f"[SKIP] {wav_path.name} | {e}")

print("Resampling completed.")
