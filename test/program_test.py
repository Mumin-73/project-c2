import torch
from TTS.api import TTS
import pyaudio
import wave
import tempfile
import os
import keyboard
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
ASSISTANT_ID = os.environ['OPENAI_ASSISTANT_ID']

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model once
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Reference audio file path for voice cloning
speaker_wav = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/TTS/TTS/tts/datasets/my_datasets/wavs/file1.wav"

# Recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

def record_audio():
    print("Recording... Press Space again to stop.")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed('space'):
            print("Recording stopped.")
            break
    stream.stop_stream()
    stream.close()
    return frames

def transcribe_audio(frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
        with wave.open(tmp_audio_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        # Read the temporary file and transcribe
        with open(tmp_audio_file.name, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="ja"
            )
        return response

def generate_response(prompt, thread_id):
    if thread_id:
        thread = client.beta.threads.get(thread_id)
    else:
        thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    response_text = ""
    for msg in messages.data:
        if msg.role == "user":
            continue
        elif msg.role == "assistant":
            for content_block in msg.content:
                if content_block.type == "text":
                    text_obj = content_block.text
                    response_text += text_obj.value
    return response_text, thread.id  # Return the thread ID along with the response text

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

if __name__ == "__main__":
    thread_id = None
    while True:
        print("Press Space to start recording.")
        keyboard.wait('space')
        frames = record_audio()
        transcribed_text = transcribe_audio(frames)
        if transcribed_text:
            print("Transcription:")
            print(transcribed_text)
            response_text, thread_id = generate_response(transcribed_text, thread_id)  # Update the thread ID
            print("Response:")
            print(response_text)
            synthesize_and_play_speech(response_text, speaker_wav, "ja")
        else:
            print("Transcription failed.")