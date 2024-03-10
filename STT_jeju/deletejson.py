import os

audio_folder = 'C:/STT_jeju/try_audio'
json_folder = 'C:/STT_jeju/try_json'

# try_json 폴더의 파일 목록 가져오기
json_files = os.listdir(json_folder)

# try_audio 폴더의 WAV 파일명 추출
wav_files = [file.split('.')[0] for file in os.listdir(audio_folder) if file.lower().endswith('.wav')]

# 삭제할 JSON 파일 찾기
files_to_delete = [json_file for json_file in json_files if json_file.split('.')[0] not in wav_files]

# 삭제 수행
for file_to_delete in files_to_delete:
    file_path = os.path.join(json_folder, file_to_delete)
    os.remove(file_path)
    print(f'Deleted: {file_path}')

print('삭제가 완료되었습니다.')
