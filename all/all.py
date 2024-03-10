import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import IPython.display as ipd
import torch
from TTS import commons
from TTS import utils
from TTS.models import SynthesizerTrn
from TTS.symbols import symbols
from TTS import text_to_sequence
import re
from TTS.k2j import korean2katakana
from TTS.j2k import japanese2korean
import soundfile as sf
import argparse
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.io import wavfile
from transformers import Wav2Vec2ForPreTraining

from VC.model.ecapa_tdnn import ECAPA_TDNN
from VC.model.generator import Generator as LVC_VC
from VC.utils.perturbations import formant_and_pitch_shift, peq, wav_to_Sound
from VC.utils.utils import (
    extract_f0_median_std,
    get_f0_norm,
    load_and_resample,
    quantize_f0_median,
    rescale_power,
)

import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO

# Voice-Other split 시작

# Customize the following options!
model = "htdemucs"
extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
two_stems = None   # only separate one stems from the rest, for instance
# two_stems = "vocals"

# Options for the output audio.
mp3 = False
mp3_rate = 320
float32 = True  # output as float 32 wavs, unsused if 'mp3' is True.
int24 = False    # output as int24 wavs, unused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!

in_path = 'base_audio'
out_path = 'seperated'

def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out

def copy_process_streams(process: sp.Popen):
    try:
        outs, errs = process.communicate(timeout=10)
        if outs:
            sys.stdout.write(outs.decode())
        if errs:
            sys.stderr.write(errs.decode())
    except sp.TimeoutExpired:
        process.kill()
        outs, errs = process.communicate()
        if outs:
            sys.stdout.write(outs.decode())
        if errs:
            sys.stderr.write(errs.decode())

def separate(inp=None, outp=None):
    inp = inp or in_path
    outp = outp or out_path
    cmd = ["python", "-m", "demucs.separate", "-o", str(outp), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]
    files = [str(f) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {in_path}")
        return
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd))
    
    # communicate 메서드 사용
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")


separate()

# STT 시작

# 저장한 모델과 관련된 요소들의 경로 설정
processor = WhisperProcessor.from_pretrained("pytorch_model",language = "Korean", task = "transcribe")
model = WhisperForConditionalGeneration.from_pretrained("pytorch_model")

# WAV 파일 경로 설정
input_wav_path = "C:/all/seperated/htdemucs/base/vocals.wav"

# 음성 파일을 읽어오기
waveform, sample_rate = torchaudio.load(input_wav_path)

# 샘플링 속도를 16000으로 변경
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# 음성 파일을 특성으로 변환
input_features = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")

# 언어 설정을 확인하고 수정
input_features["language"] = "Korean"

# 모델을 사용하여 텍스트 생성
with torch.no_grad():
    generated_ids = model.generate(**input_features)

# 생성된 텍스트 디코딩
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# TTS 시작
config_file = f"./ko.json"
model_file = f"./G_317000.pth"
device = 'cpu' # cuda:0

hps = utils.get_hparams_from_file(config_file)
isJaModel = hps.data.is_japanese_dataset
isKoModel = hps.data.is_korean_dataset

text = f"[KO]{transcription}[KO]"

text = re.sub('[\n]', '', text).strip()
if isJaModel:
    text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean2katakana(x.group(1)), text)
if isKoModel:
    text = re.sub(r'\[JA\](.*?)\[JA\]', lambda x: japanese2korean(x.group(1)), text)

print(text)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(model_file, net_g, None)

stn_tst = get_text(text, hps)

with torch.no_grad():
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()


ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=True))

output_wav_path = "generated_audio.wav"
sf.write(output_wav_path, audio, hps.data.sampling_rate)

print(f"Generated audio saved to: {output_wav_path}")

def load_ecapa_tdnn(checkpoint_path, device):
    ecapa_tdnn = ECAPA_TDNN(C=1024).eval().to(device)
    ecapa_checkpoint = torch.load(checkpoint_path, map_location=device)

    new_state_dict = {}
    for k, v in ecapa_checkpoint.items():
        if 'speaker_encoder' in k:
            key = k.replace('speaker_encoder.', '')
            new_state_dict[key] = ecapa_checkpoint[k]

    ecapa_tdnn.load_state_dict(new_state_dict)
    return ecapa_tdnn


def load_wav2vec2(device):
    wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53"
    ).eval().to(device)
    return wav2vec2


def load_lvc_vc(checkpoint_path, hp, device):
    model = LVC_VC(hp).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except KeyError:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval(inference=True)

    return model


class LVC_VC_Inference():
    def __init__(
            self,
            hp,
            lvc_vc_chkpt,
            speaker_encoder_chkpt,
            seen_speaker_emb_gmms_pkl,
            seen_speaker_f0_metadata_pkl,
            device
    ):

        # Model hyperparameters.
        self.hp = hp

        # Device.
        self.device = device

        # Define VC model components.
        self.lvc_vc = load_lvc_vc(lvc_vc_chkpt, self.hp, self.device)
        self.speaker_encoder = load_ecapa_tdnn(speaker_encoder_chkpt, self.device)
        self.wav2vec2 = load_wav2vec2(self.device)

        # Define hyperparameters.
        self.fs = self.hp.audio.sampling_rate
        self.fmin = self.hp.audio.mel_fmin
        self.fmax = self.hp.audio.mel_fmax
        self.n_mels = self.hp.audio.n_mel_channels
        self.nfft = self.hp.audio.filter_length
        self.win_length = self.hp.audio.win_length
        self.hop_length = self.hp.audio.hop_length

        # Speaker embeddings and F0 metadata for seen speakers.
        self.seen_speaker_emb_gmms = pickle.load(open(seen_speaker_emb_gmms_pkl, 'rb'))
        self.seen_speaker_f0_metadata = pickle.load(open(seen_speaker_f0_metadata_pkl, 'rb'))

    def perturb_audio(self, wav):
        # Random frequency shaping via parametric equalizer (peq).
        wav = torch.from_numpy(wav)
        wav = peq(wav, self.hp.audio.sampling_rate).numpy()

        # Formant and pitch shifting.
        sound = wav_to_Sound(wav, sampling_frequency=self.hp.audio.sampling_rate)
        sound = formant_and_pitch_shift(sound)
        perturbed_wav = sound.values[0]

        return perturbed_wav.astype(np.float32)

    def extract_features(
            self,
            source_audio,
            target_audio,
            source_seen,
            target_seen,
            source_id,
            target_id
    ):
        source_audio_tensor = torch.from_numpy(source_audio).unsqueeze(0).to(self.device)
        target_audio_tensor = torch.from_numpy(target_audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract wav2vec 2.0 features.
            wav2vec2_outputs = self.wav2vec2(source_audio_tensor, output_hidden_states=True)
            wav2vec2_feat = wav2vec2_outputs.hidden_states[12]  # (1, N, 1024)
            wav2vec2_feat = wav2vec2_feat.permute((0, 2, 1))  # (1, 1024, N)
            wav2vec2_feat = wav2vec2_feat.detach().squeeze().cpu().numpy()  # (1024, N)

        # Extract source utterance's normalized F0 contour.
        if source_seen:
            src_f0_median = self.seen_speaker_f0_metadata[source_id]['median']
            src_f0_std = self.seen_speaker_f0_metadata[source_id]['std']
        else:
            src_f0_median, src_f0_std = extract_f0_median_std(
                source_audio,
                self.fs,
                self.win_length,
                self.hop_length
            )
        f0_norm = get_f0_norm(
            source_audio,
            src_f0_median,
            src_f0_std,
            self.fs,
            int(16000*0.025),  # frame length of wav2vec 2.0
            int(16000*0.02),  # hop length of wav2vec 2.0
        )

        # Transpose to make (257, N) and crop at end to match wav2vec features.
        f0_norm = f0_norm.T[:, :wav2vec2_feat.shape[1]].astype(np.float32)

        # Extract target speaker's speaker embedding.
        if target_seen:
            target_emb = self.seen_speaker_emb_gmms[target_id].means_.astype(np.float32)[0, :]
        else:
            target_emb = self.speaker_encoder(
                target_audio_tensor, aug=False
            ).detach().squeeze().cpu().numpy()

        # Extract target speaker's quantized median F0.
        if target_seen:
            target_f0_median = self.seen_speaker_f0_metadata[target_id]['median']
        else:
            target_f0_median, _ = extract_f0_median_std(
                target_audio,
                self.fs,
                self.win_length,
                self.hop_length
            )
        target_f0_median = quantize_f0_median(target_f0_median).astype(np.float32)

        # Store all features in dictionary.
        vc_features = {
            'wav2vec2_feat': wav2vec2_feat,
            'source_f0_norm': f0_norm,
            'target_emb': target_emb,
            'target_f0_median': target_f0_median
        }

        return vc_features

    def run_inference(
            self,
            source_audio,
            target_audio,
            source_seen,
            target_seen,
            source_id,
            target_id
    ):
        # Extract all features needed for conversion.
        vc_features = self.extract_features(
            source_audio,
            target_audio,
            source_seen,
            target_seen,
            source_id,
            target_id
        )

        source_wav2vec2_feat = torch.from_numpy(vc_features['wav2vec2_feat']).unsqueeze(0)
        source_f0_norm = torch.from_numpy(vc_features['source_f0_norm']).unsqueeze(0)
        target_emb = torch.from_numpy(vc_features['target_emb']).unsqueeze(0)
        target_f0_median = torch.from_numpy(vc_features['target_f0_median']).unsqueeze(0)

        # Concatenate features to feed into model.
        noise = torch.randn(1, self.hp.gen.noise_dim, source_wav2vec2_feat.size(2)).to(self.device)
        content_feature = torch.cat((source_wav2vec2_feat, source_f0_norm), dim=1).to(self.device)
        speaker_feature = torch.cat((target_emb, target_f0_median), dim=1).to(self.device)

        # Perform conversion and rescale power to match source.
        vc_audio = self.lvc_vc(
            content_feature, noise, speaker_feature
        ).detach().squeeze().cpu().numpy()
        vc_audio = rescale_power(source_audio, vc_audio)

        return vc_audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config_wav2vec_ecapa_c32.yaml",
                    help="yaml file for configuration")
    parser.add_argument('-p', '--lvc_vc_weights', type=str,
                    default="lvc_vc_xl_vctk.pt",
                    help="path to LVC-VC model weights")
    parser.add_argument('-e', '--se_weights', type=str, default="ecapa_tdnn_pretrained.pt",
                    help="path to speaker encoder model weights")
    parser.add_argument('-g', '--gpu_idx', type=int, default=0,
                    help="index of home GPU device")
    parser.add_argument('-s', '--source_file', type=str, default="generated_audio.wav",
                    help="source utterance file")
    parser.add_argument('-t', '--target_file', type=str, default="C:/all/seperated/htdemucs/base/vocals.wav",
                    help="target utterance file")
    parser.add_argument('-o', '--output_file', type=str, default="output.wav",
                    help="output file name")
    args = parser.parse_args()

    # Select device for running models.
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")

    # Set up LVC-VC inferencer.
    hp = OmegaConf.load(args.config)
    lvc_vc_inferencer = LVC_VC_Inference(
        hp=hp,
        lvc_vc_chkpt=args.lvc_vc_weights,
        speaker_encoder_chkpt=args.se_weights,
        seen_speaker_emb_gmms_pkl='VC/metadata/ecapa_tdnn_emb_gmms_all.pkl',
        seen_speaker_f0_metadata_pkl='VC/metadata/speaker_f0_metadata.pkl',
        device=device
    )

    # Load source and target audio for conversion.
    source_audio = load_and_resample(args.source_file, hp.audio.sampling_rate)
    target_audio = load_and_resample(args.target_file, hp.audio.sampling_rate)

    # Run voice conversion and write file.
    # By setting source_seen=False and target_seen=False, we are running
    # inference as if both source and target speakers were unseen (zero-shot).
    vc_audio = lvc_vc_inferencer.run_inference(
        source_audio=source_audio,
        target_audio=target_audio,
        source_seen=False,
        target_seen=False,
        source_id=None,
        target_id=None
    )
    wavfile.write(args.output_file, hp.audio.sampling_rate, vc_audio)

    print(f"Converted audio written to {args.output_file}.")