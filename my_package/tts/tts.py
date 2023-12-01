
import requests


import time

import io
from pydub import AudioSegment
import simpleaudio as sa

def text_to_speech_and_play(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": "Bearer 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"}

    for attempt in range(5):
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        if response.status_code == 200 and response.headers.get('Content-Type') == 'audio/flac':
            audio_data = response.content
            break
        else:
            print(f"Attempt {attempt + 1} failed, error: {response.text}")
            time.sleep(5)
    else:
        print("Failed to convert text to speech after several attempts.")
        return

    # Convert FLAC to WAV
    try:
        audio_flac = AudioSegment.from_file(io.BytesIO(audio_data), format="flac")
        audio_wav = io.BytesIO()
        audio_flac.export(audio_wav, format="wav")
        audio_wav.seek(0)

        # Play the WAV audio
        wave_obj = sa.WaveObject.from_wave_file(audio_wav)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error processing or playing audio: {e}")

# Example usage
# text_to_speech_and_play("The answer to the universe is 42", "YOUR_API_KEY")