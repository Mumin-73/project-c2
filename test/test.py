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

    # Delete the temporary file
    os.remove(tmp_audio_file.name)

    # Debugging step: print the response to understand its structure
    print(response)
    
    # Check if the response is a dictionary and contains the 'text' key
    if isinstance(response, dict) and 'text' in response:
        transcription_text = response['text']
        # Print the transcription text to the console
        print("Transcription:")
        print(transcription_text)
    else:
        print("Unexpected response format:", response)

if __name__ == "__main__":
    print("Press Space to start recording.")
    keyboard.wait('space')
    frames = record_audio()
    transcribe_audio(frames)
    print("Transcription completed.")
