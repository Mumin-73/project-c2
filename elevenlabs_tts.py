import tempfile
import os
from elevenlabs import play
from elevenlabs.client import ElevenLabs

def synthesize_speech(text, api_key, voice_id):
    client = ElevenLabs(api_key=api_key)

    audio = client.generate(
            text=text,
            voice=voice_id,
            model = "eleven_multilingual_v2"
        )
    
    play(audio)