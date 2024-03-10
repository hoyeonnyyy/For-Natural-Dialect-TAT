import IPython.display as ipd
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import re
from text.k2j import korean2katakana
from text.j2k import japanese2korean
import soundfile as sf

model_name = 'ko'
config_file = f"./configs/{model_name}.json"
model_file = f"./logs/{model_name}/G_318000.pth"
device = 'cpu' # cuda:0

hps = utils.get_hparams_from_file(config_file)
isJaModel = hps.data.is_japanese_dataset
isKoModel = hps.data.is_korean_dataset

text = """
[KO]안녕암수과? 김호연이우다.. 제주도 플라팅 기술을 골아들일 거마씸.. 우도막배 탁가?, 다시 한번. 우도막배 탁가?[KO]
"""

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

output_wav_path = "generated_audio.wav"
sf.write(output_wav_path, audio, hps.data.sampling_rate)

print(f"Generated audio saved to: {output_wav_path}")