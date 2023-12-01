import openai 


def transcribe_audio(file_path='output.wav', model="whisper-1", response_format="text"):
    """
    Transcribes an audio file using OpenAI's Whisper API.
    
    :param file_path: Path to the audio file.
    :param model: The model ID to use for transcription. Default is "whisper-1".
    :param response_format: The format of the transcript output. Default is "text".
    :return: Transcribed text.
    """
    openai.api_key = "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"  # Replace with your OpenAI API key
    client = openai.OpenAI(api_key="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")
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
            print(transcript.text)
        return transcript.text  # Access the text attribute directly
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
