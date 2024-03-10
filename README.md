# flyai_gaeguri
***
# 모델 다운로드
학습된 모델들은 용량관계로 g드라이브에 넣어놨습니다

https://drive.google.com/drive/folders/1BzoyEdQS-eblhNq_pZ-FpGVQATu7_mTg?usp=drive_link

pytorch_model 폴더 : all 폴더 또는 STT_jeju 폴더에 넣기 (STT 모델)

ecapa_tdnn_pretrained.pt / lvc_vc_xl_vctk.pt 파일 : all 폴더 또는 lvc-vc의 weights 폴더에 넣기 (음성변조 모델)

G_317000.pth / G_320000.pth 파일 : all 폴더 또는 JK-VITS 폴더 또는 JK-VITS의 logs/ko 폴더에 넣기 (각각 317000,320000 steps 훈련된 TTS 모델)

모르겠으면 모든 파일 압축해놓은 flyai_finalprojectcode.Zip 다운받아서 참고하기!
***
# Meducs (음성-오디오 분리)

https://github.com/facebookresearch/demucs

여기에서 가져와서 구현하였읍니다
***
# Whisper (Speech-to-Text)

https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=121

AI-hub의 한국어 방언 발화(제주도) 데이터셋 중 200GB 사용

1. 20대 여성 화자 참여가 많아 데이터의 균형을 위해 delete20womens.py를 통해 20대 여성 화자 둘이서만 말한 데이터는 삭제

2. split30.py를 통해 Whisper 모델이 요구하는 30초에 가까운 데이터셋으로 전처리. [](){}&*과 같은 특수문자가 라벨링 데이터에 있음은 정확하지 않은 데이터이기 때문에 성능 향상을 위해 특수문자 들어간 문장은 삭제하며 전처리함. try_audio와 try_json 폴더에 각각 원천 데이터, 라벨링 데이터 저장

3. audio_preprocessing.py를 통해 원천 데이터와 라벨링 데이터를 데이터프레임 형태로 변환, 이후 16khz 샘플링, log-Mel spectrogram 변환 (음성 데이터는 용량이 너무 커서 변환하는 과정이라고 이해하면댐 검색 ㄱ), 라벨링 데이터도 변환 이후 After_preprocessing 폴더에 데이터들 저장

4. 이후 whisper_jeju.py를 통해 whisper-base, whisper-medium 등 다양하게 전이학습 해봤는데 거의 비슷했음 우리는 결국 base 썼음
***

# Vits (Text-to-Speech)

https://www.kaggle.com/datasets/bryanpark/jejueo-single-speaker-speech-dataset

카카오에서 수집한 단일화자 제주도 방언 데이터셋 (jss) 전부사용

이거는 전처리부터 학습 과정 머했는지 기억이 안남 ... train.py로 학습 돌리면 될겨
***

# LVC-VC (음성변조)

https://github.com/wonjune-kang/lvc-vc

여기에서 가져옴 

위의 jss 데이터셋이라도 학습 시키려고 preprocessing 폴더의 

resample_vctk.py

extract_f0_metadata.py

extract_all_features.py

split_data.py

preprocess_speaker_embs.py

이 순서대로 전처리 해보았는데 학습 제대로 안되고 컴퓨터도 돌리고 있는데 마감시간 부족해서 결국 못했음ㅎ 아쉽다
***
# ALL

all.py 파일 경로들 다 맞춰주면 돌아감

모델들 돌려보거나 자소서/면접 준비할때 모르는거 있으면 편하게 물어보세요^^ 
***
