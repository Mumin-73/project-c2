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

# Recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

# Audio recording function
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

# Function to transcribe audio
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

# Function to generate response from OpenAI API
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

    return response_text

if __name__ == "__main__":
    print("Press Space to start recording.")
    keyboard.wait('space')
    frames = record_audio()
    transcribed_text = transcribe_audio(frames)
    if transcribed_text:
        print("Transcription:")
        print(transcribed_text)
        response_text = generate_response(transcribed_text, None)
        print("Response:")
        print(response_text)
    else:
        print("Transcription failed.")