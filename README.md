```markdown
# Cut and Paste 기반 이미지 증강 (Python)

본 프로젝트는 Python과 OpenCV를 사용하여 Cut and Paste 기법을 활용한 이미지 증강을 구현합니다. 정상 이미지에 결함 이미지를 무작위 변환(이동, 회전)하여 합성하고, 증강된 이미지와 그에 해당하는 마스크를 생성합니다. 이는 결함 데이터가 부족한 환경에서 모델 학습을 위한 추가 데이터를 확보하는 데 유용합니다.

## 주요 기능

- **결함 합성:** 정상 이미지에 결함 이미지를 다양한 변환(이동, 회전)을 적용하여 합성합니다.
- **경계 제약:** 합성된 결함이 주어진 경계 마스크 내에 있도록 제약합니다.
- **무작위 변환:** 이동 및 회전 변환량을 무작위로 설정하여 다양한 증강 효과를 생성합니다.
- **마스크 생성:** 증강된 이미지와 함께 해당 결함 영역을 나타내는 마스크를 생성합니다.
- **결과 저장:** 생성된 증강 이미지와 마스크를 날짜 및 시간별로 자동 생성되는 결과 디렉토리에 저장합니다.
- **tqdm 진행 표시:** 이미지 생성 과정을 시각적으로 확인할 수 있도록 진행률 바를 제공합니다.

## 요구 사항

- Python 3.x
- OpenCV (`cv2`) 라이브러리
- NumPy (`numpy`) 라이브러리
- tqdm 라이브러리
- os 모듈
- datetime 모듈

## 설치

Python 3.x가 시스템에 설치되어 있는지 확인하십시오. 그 후, 필요한 라이브러리를 pip를 사용하여 설치할 수 있습니다.

```bash
pip install opencv-python numpy tqdm
```

## 디렉토리 구조

프로젝트 디렉토리는 다음과 같은 구조를 가질 수 있습니다.

```
your_project_directory/
├── your_script_name.py  # 메인 스크립트 파일
├── image_dir/             # 입력 이미지 및 마스크를 저장하는 디렉토리 (선택 사항)
│   ├── 000_normal.png   # 정상 이미지 예시
│   ├── 000_mask_1.bmp   # 경계 마스크 예시
│   ├── 000_cut.png      # 결함 이미지 예시
│   └── 000_mask_cut.png # 결함 마스크 예시
└── results/             # 결과 이미지 및 마스크가 저장될 디렉토리 (스크립트 실행 시 자동 생성)
```

## 사용법

1.  **스크립트 파일 (`your_script_name.py`)을 저장합니다.** (위 코드 내용을 파일에 저장)
2.  **입력 이미지 및 마스크를 준비합니다.**
    -   정상 이미지 (`000_normal.png`)
    -   경계 마스크 (`000_mask_1.bmp`): 결함이 나타날 수 있는 유효 영역을 나타내는 마스크 (흰색 영역).
    -   결함 이미지 (`000_cut.png`): 붙여넣을 결함 영역 이미지.
    -   결함 마스크 (`000_mask_cut.png`): 결함 영역을 나타내는 그레이스케일 마스크 (흰색 영역).
3.  **스크립트 내 이미지 경로 (`image_dir`)를 실제 이미지 파일이 있는 디렉토리 경로로 수정합니다.** 필요하다면 각 이미지 파일 이름도 확인하여 수정합니다.
4.  **증강 파라미터 (`augmentation_params`)를 조정합니다.**
    -   `min_x_offset`, `max_x_offset`: x축 이동 범위
    -   `min_y_offset`, `max_y_offset`: y축 이동 범위
    -   `min_rotation`, `max_rotation`: 회전 각도 범위
5.  **생성하고 싶은 증강 이미지의 개수 (`num_augmentations`)를 `paste_defect_on_normal_with_constraints` 함수 호출 시 설정합니다.**
6.  **터미널 또는 명령 프롬프트에서 스크립트를 실행합니다.**

    ```bash
    python your_script_name.py
    ```

## 결과 확인

스크립트가 실행되면 현재 날짜와 시간을 기준으로 `results` 디렉토리 내에 새로운 폴더 (`YYYYMMDD_HHMMSS_result`)가 생성됩니다. 이 폴더에는 생성된 증강 이미지 (`augmented_image_0.png`, `augmented_image_1.png`, ...)와 해당 마스크 (`augmented_mask_0.png`, `augmented_mask_1.png`, ...)가 저장됩니다. 마스크 이미지는 결함 영역을 흰색으로 나타내는 그레이스케일 이미지입니다.

## 파라미터 조정

-   **`augmentation_params`**: 이동 및 회전 범위를 조절하여 생성되는 증강 데이터의 다양성을 제어할 수 있습니다.
-   **`num_augmentations`**: 생성할 증강 이미지의 총 개수를 설정합니다.
-   **경계 마스크 (`boundary_mask`)**: 결함이 나타날 수 있는 영역을 제한하여 현실적인 증강 데이터를 생성하는 데 중요한 역할을 합니다.

이 코드를 통해 Cut and Paste 기반의 이미지 증강을 수행하고, 학습 데이터셋을 확장하는 데 활용할 수 있습니다.
```
