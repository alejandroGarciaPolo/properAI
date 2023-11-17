import json
import requests
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pyaudio
import wave
import speech_recognition as sr
import pickle
import time
import simpleaudio as sa
import io



openai.api_key = ""  # Replace with your OpenAI API key
client = openai.OpenAI(api_key="")


class ProjectNote:
    def __init__(self, main_idea, steps=None):
        self.main_idea = main_idea
        self.steps = steps if steps is not None else []

    def add_step(self, step):
        self.steps.append(step)

    def __str__(self):
        return f"Idea: {self.main_idea}\nSteps: {'; '.join(self.steps)}"

def load_note_bank(filename="note_bank.pkl"):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError):
        return []  # Return an empty list if file doesn't exist or is empty

def save_note_bank(note_bank, filename="note_bank.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(note_bank, file)



note_bank = load_note_bank()


# print(note_bank[0].main_idea)

def create_new_note(main_idea):
    note = ProjectNote(main_idea)
    note_bank.append(note)
    return "Note created."

def add_step_to_last_note(step):
    if note_bank:
        note_bank[-1].add_step(step)
        return "Step added."
    else:
        return "No note to add step to."

def get_all_notes():
    return "\n\n".join(str(note) for note in note_bank)


def test_function(is_testing):
    print('it worked', is_testing)


def record_until_silence(pause_threshold=3.0):
    """
    Records audio from the microphone until silence is detected.
    
    :param pause_threshold: Minimum length of silence (in seconds) at the end of a phrase.
    """
    r = sr.Recognizer()
    r.pause_threshold = pause_threshold  # Set the threshold for pause

    with sr.Microphone() as source:
        print("Please start speaking. Recording will stop after a pause of " + str(pause_threshold) + " seconds.")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    # Save the audio to a WAV file
    with open("output.wav", "wb") as f:
        f.write(audio.get_wav_data())

    print("Recording stopped and saved to 'output.wav'")
# def record_until_silence(timeout=None, phrase_time_limit=None):
#     """
#     Records audio from the microphone until silence is detected.
    
#     :param timeout: Maximum time to wait for speech. If None, wait indefinitely.
#     :param phrase_time_limit: Maximum time for a single phrase. Set to 5 seconds for your use case.
#     """
#     r = sr.Recognizer()

#     with sr.Microphone() as source:
#         print("Please start speaking. Recording will stop after a 5-second pause.")
#         r.adjust_for_ambient_noise(source)
#         # Listen for the first phrase and extract audio
#         audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

#     # Save the audio to a WAV file
#     with open("output.wav", "wb") as f:
#         f.write(audio.get_wav_data())

#     print("Recording stopped and saved to 'output.wav'")

def record_audio(output_filename, record_seconds=7, chunk=1024, format=pyaudio.paInt16, channels=1, rate=44100):
    """
    Records audio from the microphone and saves it to a file.
    
    :param output_filename: The name of the file to save the recording.
    :param record_seconds: Duration of the recording in seconds. Default is 5 seconds.
    :param chunk: Number of frames in the buffer.
    :param format: Sample format.
    :param channels: Number of channels.
    :param rate: Sampling rate.
    """
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("* recording")

    frames = []

    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Example usage



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


def transcribe_audio(file_path='output.wav', model="whisper-1", response_format="text"):
    """
    Transcribes an audio file using OpenAI's Whisper API.
    
    :param file_path: Path to the audio file.
    :param model: The model ID to use for transcription. Default is "whisper-1".
    :param response_format: The format of the transcript output. Default is "text".
    :return: Transcribed text.
    """

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
# Example usage


def create_function_json():
    function = {}
    
    function['name'] = input("Enter the function name: ")
    function['description'] = input("Enter the function description: ")

    # Defining parameter types that can be chosen
    available_types = ['string', 'integer', 'boolean']

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    while True:
        param_name = input("Enter parameter name (or press enter to finish): ")
        if not param_name:
            break

        param_type = ""
        while param_type not in available_types:
            param_type = input(f"Enter parameter type ({'/'.join(available_types)}): ")

        param_desc = input("Enter parameter description: ")

        parameters['properties'][param_name] = {
            "type": param_type,
            "description": param_desc
        }

        if input("Is this parameter required? (yes/no): ").lower() == 'yes':
            parameters['required'].append(param_name)

    function['parameters'] = parameters

    # Printing the function JSON
    print("\nGenerated Function JSON:")
    print(json.dumps(function, indent=4))

# Add more functions as needed
available_functions = {
    "test_function": test_function,
    "create_function_json": create_function_json,
    'create_new_note': create_new_note,


}

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def gpt_chat_and_execute(question, context=None, functions=None, model="gpt-3.5-turbo-0613", function_call=None):
    # Send request to GPT
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    
    messages = [{"role": "user", "content": question}]
    if context:
        for message in context:
            messages.append({"role": "system", "content": message})

    json_data = {"model": model, "messages": messages}
    
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return None

    # Execute function from response
    try:
        assistant_message = response.json()["choices"][0]["message"]
        # print(assistant_message)
        if 'function_call' in assistant_message:
            function_call = assistant_message['function_call']
            function_name = function_call['name']
            function_args = json.loads(function_call['arguments'])

            if function_name in available_functions:
                return available_functions[function_name](**function_args)
            else:
                raise ValueError(f"Function {function_name} not defined.")
        else:
            print(assistant_message['content'])
            return assistant_message['content']
    except Exception as e:
        print(f"Error executing function: {e}")
        return None

# Example usage
question = "Hey please add this two numbers, 5 and 7"

functions = [
{
    "name": "test_function",
    "description": "This is a testng function solely used to test your function calling ability, use it when requested",
    "parameters": {
        "type": "object",
        "properties": {
            "is_testing": {
                "type": "boolean",
                "description": "set to true when called since you would be testing"
            }
        },
        "required": [
            "is_testing"
        ]
    }
},
{
    "name": "create_function_json",
    "description": "This is a function that we can use to prompt the user to create a schema for a missing function",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
},
{
    "name": "create_new_note",
    "description": "Function to create a Project Note",
    "parameters": {
        "type": "object",
        "properties": {
            "main_idea": {
                "type": "string",
                "description": "the main idea to execute"
            }
        },
        "required": [
            "main_idea"
        ]
    }
},

]

# gpt-4-1106-preview

# record_audio("output.wav")
# # transcribed=transcribe_audio()

# print('out')
record_until_silence()
text_to_speech_and_play(gpt_chat_and_execute(question=transcribe_audio(), functions=functions, function_call='auto'))

save_note_bank(note_bank)