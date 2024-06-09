import torch
import pyaudio
import wave
import tempfile
import os
import keyboard
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs_tts import synthesize_speech

# Load environment variables
load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
ASSISTANT_ID = os.environ['OPENAI_ASSISTANT_ID']
ELEVENLABS_API = os.environ['ELEVENLABS_API']
VOICE_ID = os.environ['VOICE_ID']


# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def generate_response(prompt, thread_id=None):
    if thread_id is None:
        thread = client.beta.threads.create()
        thread_id = thread.id
    else:
        thread = client.beta.threads.retrieve(thread_id)

    # Delete previous assistant messages
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == "assistant":
            client.beta.threads.messages.delete(thread_id=thread_id, message_id=msg.id)

    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    )

    response_text = ""
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        for msg in messages.data:
            if msg.role == "assistant":
                for content_block in msg.content:
                    if content_block.type == "text":
                        response_text = content_block.text.value
                        break
    else:
        print(f"Run status: {run.status}")

    return response_text, thread_id


def synthesize_and_play_speech(text):
    synthesize_speech(text, ELEVENLABS_API, VOICE_ID)

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
            synthesize_and_play_speech(response_text)
            response_text = ""
        else:
            print("Transcription failed.") 