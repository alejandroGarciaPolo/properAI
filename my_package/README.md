# MyPackage

MyPackage is a Python package for audio processing, text-to-speech conversion, and transcription using various APIs including OpenAI's GPT models. This package allows users to record audio, play back audio, convert text to speech, and transcribe audio files.

## Installation

You can install MyPackage using pip:

pip install MyPackage
Features
Audio Recording: Record audio through a microphone and save it as a WAV file.
Audio Playback: Play audio files.
Text-to-Speech: Convert text to speech using Hugging Face's API.
Transcription: Transcribe audio files using OpenAI's Whisper API.
API Interaction: Functions for interacting with OpenAI's GPT models and other APIs.
## Configuration

Before using MyPackage, set your OpenAI API key:

```python
import my_package.api_interaction as api

api.Config.set_api_key("your_api_key_here")

Usage

Here's a quick example of how to record and transcribe audio:


from MyPackage.audio.recording import record_until_silence
from MyPackage.transcription.transcription import transcribe_audio

# Record audio
record_until_silence()

# Transcribe the recorded audio
transcript = transcribe_audio(file_path="output.wav")
print(transcript)
For more examples and usage, please refer to the Documentation.

Contributing:

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.

Contact
agarciap@uwaterloo.ca

Project Link: https://github.com/maestromaximo/MyPackage



