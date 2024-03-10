from pydub import AudioSegment
import json
import os
'''
def split_audio_and_json(json_file_path, wav_file_path, output_directory):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    audio = AudioSegment.from_wav(wav_file_path)
    
    split_index = 1
    segment_start_time = 0  # 분할 시작 시간
    last_end_time = 0  # 마지막으로 처리된 발화의 종료 시간
    segment_utterances = []  # 현재 분할에 포함된 발화들
    
    for utterance in data['utterance']:
        u_start = int(utterance['start'] * 1000)  # 발화 시작 시간 (밀리초 단위)
        u_end = int(utterance['end'] * 1000)  # 발화 종료 시간 (밀리초 단위)
        
        # 현재 분할의 길이가 30초를 초과할 예정이면 분할을 완료하고 새 분할을 시작
        if u_end - segment_start_time > 30000:
            # 오디오 분할 및 저장
            split_audio = audio[segment_start_time:last_end_time]
            split_audio_path = os.path.join(output_directory, f"{data['id']}_split_{split_index}.wav")
            split_audio.export(split_audio_path, format="wav")
            
            # JSON 분할 및 저장
            split_json_path = os.path.join(output_directory, f"{data['id']}_split_{split_index}.json")
            with open(split_json_path, 'w', encoding='utf-8') as json_out:
                json.dump({
                    'id': data['id'],
                    'metadata': data['metadata'],
                    'speaker': data['speaker'],
                    'setting': data['setting'],
                    'utterance': segment_utterances
                }, json_out, indent=4, ensure_ascii=False)
            
            split_index += 1
            segment_start_time = last_end_time  # 새 분할의 시작 시간을 마지막 발화의 종료 시간으로 설정
            segment_utterances = []

        last_end_time = u_end
        segment_utterances.append(utterance)
    
    # 마지막 분할 처리
    if segment_utterances:
        split_audio = audio[segment_start_time:last_end_time]
        split_audio_path = os.path.join(output_directory, f"{data['id']}_split_{split_index}.wav")
        split_audio.export(split_audio_path, format="wav")
        
        split_json_path = os.path.join(output_directory, f"{data['id']}_split_{split_index}.json")
        with open(split_json_path, 'w', encoding='utf-8') as json_out:
            json.dump({
                'id': data['id'],
                'metadata': data['metadata'],
                'speaker': data['speaker'],
                'setting': data['setting'],
                'utterance': segment_utterances
            }, json_out, indent=4, ensure_ascii=False)

def process_files(json_directory, audio_directory, output_directory):
    for json_file in os.listdir(json_directory):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_directory, json_file)
            base_name = os.path.splitext(json_file)[0]
            wav_file = f"{base_name}.wav"
            wav_path = os.path.join(audio_directory, wav_file)
            if os.path.exists(wav_path):
                split_audio_and_json(json_path, wav_path, output_directory)
            else:
                print(f"Audio file not found for {json_file}")

# 디렉토리 경로 설정
json_file_path = 'C:/STT_jeju/try_json'  # JSON 파일 경로
audio_file_path = 'C:/STT_jeju/try_audio'  # 오디오 파일 경로
output_directory = 'C:/STT_jeju/audio_after'  # 출력 디렉토리
os.makedirs(output_directory, exist_ok=True)


# 디렉토리 내의 파일 처리
process_files(json_file_path, audio_file_path, output_directory)
'''


def filter_and_split_audio(json_path, audio_path, output_directory, segment_length=30000):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    audio = AudioSegment.from_file(audio_path)

    # 조건을 만족하는 발화와 해당 오디오 세그먼트를 필터링
    filtered_utterances = []
    segments = []
    current_segment = []
    current_length = 0

    for utterance in data['utterance']:
        dialect_form = utterance['dialect_form']
        # 괄호, '&', '*' 중 하나라도 포함되지 않는 발화만 필터링
        if not any(char in dialect_form for char in ['[', ']', '(', ')', '{', '}', '&', '*']):
            start_ms = int(utterance['start'] * 1000)
            end_ms = int(utterance['end'] * 1000)
            segment_duration = end_ms - start_ms

            # 현재 세그먼트 길이 확인 및 30초 근방에서 새 세그먼트 시작
            if current_length + segment_duration > segment_length:
                segments.append((current_segment, current_length))
                current_segment = []
                current_length = 0

            current_segment.append(utterance)
            current_length += segment_duration

    # 마지막 세그먼트 추가
    if current_segment:
        segments.append((current_segment, current_length))

    # 각 세그먼트에 대한 오디오 및 JSON 파일 생성
    for index, (segment_utterances, _) in enumerate(segments, start=1):
        segment_audio = sum(audio[int(u['start'] * 1000):int(u['end'] * 1000)] for u in segment_utterances)
        segment_audio_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(json_path))[0]}_segment_{index}.wav")
        segment_audio.export(segment_audio_path, format='wav')

        segment_json_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(json_path))[0]}_segment_{index}.json")
        with open(segment_json_path, 'w', encoding='utf-8') as segment_json_file:
            json.dump({
                'id': f"{data['id']}_segment_{index}",
                'metadata': data['metadata'],
                'speaker': data['speaker'],
                'setting': data['setting'],
                'utterance': segment_utterances
            }, segment_json_file, ensure_ascii=False, indent=4)

# 경로 설정
json_directory = 'C:/STT_jeju/try_json'
audio_directory = 'C:/STT_jeju/try_audio'
output_directory = 'C:/STT_jeju/audio_after'
os.makedirs(output_directory, exist_ok=True)

# 파일 처리
for json_file in os.listdir(json_directory):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_directory, json_file)
        audio_path = os.path.join(audio_directory, os.path.splitext(json_file)[0] + '.wav')
        if os.path.exists(audio_path):
            filter_and_split_audio(json_path, audio_path, output_directory)