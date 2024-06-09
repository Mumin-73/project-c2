import torch
from TTS.api import TTS
import pyaudio
import wave
import tempfile
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model once
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Reference audio file path for voice cloning
speaker_wav = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/wavs/file1.wav"

def synthesize_and_play_speech(text, speaker_wav, language):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # Synthesize speech from the input text using the speaker reference audio
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=temp_file.name)
        
        # Play the synthesized speech using PyAudio
        wf = wave.open(temp_file.name, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        
        temp_file.close()
        
        # Delete the temporary file
        os.unlink(temp_file.name)

while True:
    # Prompt user for input text
    text = input("텍스트를 입력하세요 (종료하려면 'exit' 입력): ")
    if text.lower() == 'exit':
        break
    
    # Synthesize and play speech
    synthesize_and_play_speech(text, speaker_wav, "ja")