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
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from transformers import WhisperProcessor
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from datasets import load_dataset
from datasets import load_from_disk

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

path = "C:/STT_jeju/audio_after/*.wav"
raw_data_list = glob.glob(path, recursive=True)
raw_data_list = sorted(raw_data_list)

path2 = "C:/STT_jeju/audio_after/*"
labeled_data_list = glob.glob(path2)
labeled_data_list = sorted([file for file in labeled_data_list if file.endswith(".txt")])

transcript_list = []

for labeled_data in tqdm(labeled_data_list):
    with open(labeled_data, 'r', encoding='UTF8') as f:
        line = f.readline()
        transcript_list.append(line)

df = pd.DataFrame(data=transcript_list, columns = ["transcript"])

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

local_dataset_path = "C:/STT_jeju/After_Preprocessing"
low_call_voices_prepreocessed = load_from_disk(local_dataset_path)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load('cer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="C:/STT_jeju/whisper_second_model",  # 로컬에 저장할 디렉토리 경로
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # 배치 크기가 2배 감소할 때마다 2배씩 증가
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=6000,  # epoch 대신 설정
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",  # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=low_call_voices_prepreocessed["train"],
    eval_dataset=low_call_voices_prepreocessed["valid"],  # or "test"
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

trainer.evaluate()

# 트레이너에서 학습한 모델을 로컬에 저장
trainer.save_model("C:/STT_jeju/whisper_second_model")

# Feature extractor, processor, tokenizer도 로컬에 저장
processor.save_pretrained("C:/STT_jeju/whisper_second_model")
feature_extractor.save_pretrained("C:/STT_jeju/whisper_second_model")
tokenizer.save_pretrained("C:/STT_jeju/whisper_second_model")