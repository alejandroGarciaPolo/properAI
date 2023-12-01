
import pyaudio
import wave
import speech_recognition as sr


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
