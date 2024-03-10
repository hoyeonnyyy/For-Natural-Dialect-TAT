import os
import json
import numpy as np
import matplotlib.pyplot as plt

# JSON 파일이 있는 폴더 경로 설정
folder_path = "C:/STT_jeju/try_json"

# WAV 파일이 있는 폴더 경로 설정
wav_folder_path = "C:/STT_jeju/try_audio"

# 폴더 내의 모든 JSON 파일 검색 및 데이터 처리를 위한 리스트 초기화
sexs = []
ages = []
sexs_ages = []
dialect_sentences = []
women_20 = []

# 폴더 내의 모든 JSON 파일 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        # JSON 파일 불러오기
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # 여성이면서 20대인 발화자가 있는지 확인하기 위한 플래그
        is_woman_20 = False

        for speaker in data['speaker']:
            sex = speaker['sex']
            age = int((speaker['age'])[:2])

            if sex:
                sexs.append(sex)
                ages.append(age)
                sexs_ages.append([sex, age])

            # 모든 발화자가 여성이면서 20대인지 확인
            if sex == '여성' and age == 20:
                is_woman_20 = True
            else:
                # 한 명이라도 조건에 맞지 않으면 False로 설정
                is_woman_20 = False
                break  # 한 명이라도 조건에 맞지 않으면 더 이상 확인할 필요 없음

        # 해당 파일의 모든 발화자가 여성이면서 20대인 경우만 women_20에 추가
        if is_woman_20:
            women_20.append(data['id'])

# WAV 파일 삭제
for file_name in women_20:
    # JSON 파일명에 확장자 ".json"을 추가
    json_file_name_with_extension = file_name + ".json"
    json_file_path = os.path.join(folder_path, json_file_name_with_extension)

    # WAV 파일명은 JSON 파일명과 동일하다고 가정
    wav_file_name_with_extension = file_name + ".wav"
    wav_file_path = os.path.join(wav_folder_path, wav_file_name_with_extension)

    # JSON 파일 삭제
    if os.path.exists(json_file_path):
        os.remove(json_file_path)
        print(f"{json_file_name_with_extension}를 삭제했습니다.")
    else:
        print(f"{json_file_name_with_extension}가 존재하지 않습니다.")

    # WAV 파일 삭제
    if os.path.exists(wav_file_path):
        os.remove(wav_file_path)
        print(f"{wav_file_name_with_extension}를 삭제했습니다.")
    else:
        print(f"{wav_file_name_with_extension}가 존재하지 않습니다.")