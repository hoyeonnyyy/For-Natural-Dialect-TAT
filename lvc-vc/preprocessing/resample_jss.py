import os
import sys

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append("../")
from utils.utils import load_and_resample

fs = 16000
wav_dir = "C:/lvc-vc/jss"
resample_dir = "C:/lvc-vc/jss_resample"

os.makedirs(resample_dir, exist_ok=True)

# Process all files in the given directory.
file_list = sorted(os.listdir(wav_dir))
for file_name in file_list:
    print(f"Processing {file_name}...")

    wav_path = os.path.join(wav_dir, file_name)

    # Load and resample time domain signal.
    resampled_float = load_and_resample(wav_path, fs)
    resampled_int = (resampled_float * 32768.).astype('int16')

    save_path = os.path.join(resample_dir, file_name)
    wavfile.write(save_path, fs, resampled_int)