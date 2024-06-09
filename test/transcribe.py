from openai import OpenAI
import os
import wave
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

def transcribe_audio(frames, sample_width, rate, channels):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
        with wave.open(tmp_audio_file.name, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        # 임시 파일 읽기 및 전사
        with open(tmp_audio_file.name, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json",  # 응답 형식을 JSON으로 설정
                language="ja"
            )

    # 임시 파일 삭제
    os.remove(tmp_audio_file.name)

    # Debugging step: print the response to understand its structure
    print(response)

    # 응답이 Transcription 객체라고 가정하고 텍스트 추출
    if hasattr(response, 'text'):
        return response.text
    else:
        raise ValueError(f"Unexpected response format: {response}")