from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'json', 
        'requests', 
        'openai', 
        'tenacity',  
        'pyaudio', 
        'wave', 
        'speech_recognition', 
        'pickle', 
        'time', 
        'simpleaudio', 
        'io', 
        'pydub',  
        'simpleaudio'
    ],
)
