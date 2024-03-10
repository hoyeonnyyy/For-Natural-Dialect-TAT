from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from datasets import Audio
from datasets import load_dataset

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

path = "C:/STT_jeju/audio_after/*.wav"
raw_data_list = glob.glob(path, recursive=True)
raw_data_list = sorted(raw_data_list)

path2 = "C:/STT_jeju/audio_after/*"
labeled_data_list = glob.glob(path2)
labeled_data_list = sorted([file for file in labeled_data_list if file.endswith(".txt")])

transcript_list = []

for labeled_data in tqdm(labeled_data_list):
    with open(labeled_data, 'r', encoding='UTF8') as f:
        lines = f.readlines()
        for line in lines:
            # Remove patterns like "1:", "2:", "3:", etc.
            cleaned_line = re.sub(r'\b\d+:\s*', '', line)
            transcript_list.append({"transcript": cleaned_line.strip()})

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(transcript_list, columns=["transcript"])

df["raw_data"] = raw_data_list

ds = Dataset.from_dict({"audio": [path for path in df["raw_data"]],
                       "transcripts": [transcript for transcript in df["transcript"]]}).cast_column("audio", Audio(sampling_rate=16000))
print(df)