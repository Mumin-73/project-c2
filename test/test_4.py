import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Function to transcribe audio file and save as txt
def transcribe_audio_file(file_path, output_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="ja"
        )

    transcription_text = response

    # Save the transcription text to a file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(transcription_text)

if __name__ == "__main__":
    # Specify the path to your WAV file
    wav_file_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/origin_file.wav"

    # Specify the output path for the transcription file
    output_file_path = "C:/Users/wjdgy/OneDrive - 계명대학교/바탕 화면/문서/Coding/Python/C.C project/transcribed_text.txt"

    # Check if the WAV file exists
    if os.path.isfile(wav_file_path):
        # Transcribe the audio file and save the output
        transcribe_audio_file(wav_file_path, output_file_path)
        print("Transcription completed.")
    else:
        print("File not found:", wav_file_path)