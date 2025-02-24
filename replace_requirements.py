# 로컬 경로에서 버전 정보만 추출하여 requirements.txt 파일 만들기

import re

# 기존 requirements.txt 파일 읽기
with open('requirements.txt', 'r') as file:
    lines = file.readlines()

# 버전 정보만 남기기 위한 패턴 (예시: 패키지==버전 형식)
new_lines = []
for line in lines:
    # 로컬 경로를 포함하는 라인만 필터링
    if 'file://' in line:
        # 파일 경로에서 버전 정보만 추출 (패키지 이름 + 버전)
        match = re.search(r'([a-zA-Z0-9_-]+)==([0-9.]+)', line)
        if match:
            new_lines.append(f"{match.group(1)}=={match.group(2)}\n")
    else:
        new_lines.append(line)

# 새로운 requirements.txt 파일로 저장
with open('requirements_new.txt', 'w') as file:
    file.writelines(new_lines)

print("변경된 requirements_new.txt 파일 생성 완료!")