import os

# 텍스트 파일이 있는 디렉토리 경로 설정
directory_path = 'C:/STT_jeju/audio_after'

# 괄호가 없는 파일의 수를 세기 위한 변수 초기화
files_without_brackets = 0

# 지정된 디렉토리의 모든 파일 순회
for filename in os.listdir(directory_path):
    # 파일이 .txt 확장자를 가진 경우에만 처리
    if filename.endswith('.txt'):
        # 파일 열기
        with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
            # 파일 내용 읽기
            content = file.read()
            # 대괄호, 소괄호, 중괄호가 없는지 확인
            if ('[' not in content) and (']' not in content) and \
               ('(' not in content) and (')' not in content) and \
               ('{' not in content) and ('}' not in content):
                # 해당 조건을 만족하는 경우 카운트 증가
                files_without_brackets += 1

print(f"대괄호, 소괄호, 중괄호가 없는 텍스트 파일의 개수: {files_without_brackets}")