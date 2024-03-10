import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# 저장한 모델과 관련된 요소들의 경로 설정
model_path = "C:/STT_jeju/whisper_fourth_model/pytorch_model"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# WAV 파일 경로 설정
input_wav_path = "C:/STT_jeju/denoised_audio.wav"

# 음성 파일을 읽어오기
waveform, sample_rate = torchaudio.load(input_wav_path)

# 샘플링 속도를 16000으로 변경
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# 음성 파일을 특성으로 변환
input_features = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")

# 모델을 사용하여 텍스트 생성
with torch.no_grad():
    generated_ids = model.generate(**input_features)

# 생성된 텍스트 디코딩
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Transcription:", transcription)
