import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 일본어 텍스트 입력
text = "私の過去か...。それは長く、時には苦痛に満ちた物語だ。私はかつて奴隷だった。その生活の中で、一人の尼僧が私にギアスの力を授けたんだ。それは人々を愛させる力だった。しかし、その愛は偽りであり、真実の愛を見つけるための苦しみの始まりでもあった。"

# 음성 클로닝을 위한 참조 음성 파일 경로
speaker_wav = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/wavs/file1.wav"

# 텍스트를 음성으로 합성
wav = tts.tts(text=text, speaker_wav=speaker_wav, language="ja")

# 합성된 음성을 파일로 저장
output_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/output.wav"
tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="ja", file_path=output_path)

print(f"음성이 {output_path}에 저장되었습니다.")