1) conda prompt 실행
2) prompt 내에서 nexus_env.yaml 파일이 있는 폴더로 이동
3) 명령어 입력(nexus라는 이름의 가상환경 생성. 2022.11.17 기준 python 3.9 등 SW가 잘 작동하는 package 버전 포함): conda env create -f nexus_env.yaml
 3-1) 가상환경의 이름을 바꾸고 싶으면 yaml 파일의 key인 'name'의 value 변경
4) 명령어 입력(nexus라는 가상환경을 가동하고 그 위에서 프로그램 실행) conda activate nexus 
