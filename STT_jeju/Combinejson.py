import json
import os

# JSON 파일이 있는 폴더 경로 설정
folder_path = 'C:/STT_jeju/audio_after'  # 폴더 경로를 적절하게 변경하세요

# 폴더 내의 모든 JSON 파일 검색
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # JSON 파일 불러오기
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # "dialect_form"만 추출하여 문장으로 만들기
        dialect_sentences = []
        for utterance in data['utterance']:
            dialect_form = utterance['dialect_form']
            if dialect_form:
                dialect_sentences.append(dialect_form)

        # 추출한 "dialect_form"을 한 문장으로 합치기
        dialect_text = ' '.join(dialect_sentences)

        # 결과를 TXT 파일로 저장
        txt_filename = filename.replace('.json', '.txt')
        txt_filepath = os.path.join(folder_path, txt_filename)
        with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(dialect_text)