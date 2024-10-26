import pyttsx3
try:
    from dataloader import load_config
except ImportError:
    try:
        from .dataloader import load_config
    except ImportError:
        from src.dataloader import load_config

config = load_config()


def speak(text):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    voice_type = config['voice']['type']
    voice_index = 0 if voice_type == 'male' else 1
    engine.setProperty('voice', voices[voice_index].id)
    engine.say(text)
    engine.runAndWait()
