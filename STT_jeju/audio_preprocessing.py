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

train_testvalid = ds.train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
datasets = DatasetDict({
    "train": train_testvalid["train"],
    "test": test_valid["test"],
    "valid": test_valid["train"]})

low_call_voices  = datasets

def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # target text를 label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    return batch

low_call_voices = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=None)

local_save_path = "C:/STT_jeju/After_Preprocessing"
low_call_voices.save_to_disk(local_save_path)