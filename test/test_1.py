import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model once
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Reference audio file path for voice cloning
speaker_wav = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/wavs/file1.wav"

def synthesize_speech(text, speaker_wav, language, output_path):
    # Synthesize speech from the input text using the speaker reference audio
    wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
    # Save the synthesized speech to a file
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
    print(f"음성이 {output_path}에 저장되었습니다.")

while True:
    # Prompt user for input text
    text = input("텍스트를 입력하세요 (종료하려면 'exit' 입력): ")
    if text.lower() == 'exit':
        break
    # Define output path
    output_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/output.wav"
    # Synthesize and save speech
    synthesize_speech(text, speaker_wav, "ja", output_path)
