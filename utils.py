import os
import shutil
import subprocess
from multiprocessing import Pool
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path_dir):
    sound = AudioSegment.from_mp3(mp3_path)
    filename = os.path.splitext(os.path.basename(mp3_path))[0]
    output_path = os.path.join(wav_path_dir, filename + ".wav")
    sound.export(output_path, format="wav")
    return output_path

def convert_stereo_to_mono(wav_path, mono_path_dir):
    sound = AudioSegment.from_wav(wav_path)
    sound = sound.set_channels(1)
    filename = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(mono_path_dir, filename + "_mono.wav")
    sound.export(output_path, format="wav")
    return output_path