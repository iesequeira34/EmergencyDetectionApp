import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(seconds: int, fs: int = 16000, channels: int = 1, output_file: str='output.wav'):
    print(f"\nSpeak for {seconds}s...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
    sd.wait()
    write(output_file, fs, myrecording)
    return output_file
