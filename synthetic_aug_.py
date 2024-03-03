import cv2
import numpy as np
import random
from tqdm import tqdm
import os
from datetime import datetime

## 240303 commit

def create_results_directory(base_path=""):
    now = datetime.now()
    directory_name = now.strftime("%Y%m%d_%H%M%S_result")
    results_path = os.path.join(base_path, directory_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path

def is_within_boundary(translated_mask, boundary_mask):
    # 경계 내에 있는지 체크하는 함수
    return np.all((translated_mask & boundary_mask) == translated_mask)

def adjust_mask_within_boundary(defect_mask, boundary_mask):
    # 경계를 벗어나는 부분 제거
    return cv2.bitwise_and(defect_mask, boundary_mask)

def paste_defect_on_normal_with_constraints(normal_image, defect_image, defect_mask, boundary_mask, operation, params, num_augmentations):
    result_images = []
    result_masks = []  # 변형된 마스크를 저장할 리스트 추가

    for _ in tqdm(range(num_augmentations), desc="Generating images"):
        result_image = normal_image.copy()
        current_defect_image = defect_image
        current_defect_mask = defect_mask

        # Translation 적용
        if operation == 'translate' or operation == 'both':
            x_offset = random.randint(params['min_x_offset'], params['max_x_offset'])
            y_offset = random.randint(params['min_y_offset'], params['max_y_offset'])
            current_defect_image = translate_image(defect_image, x_offset, y_offset, result_image.shape[1], result_image.shape[0])
            current_defect_mask = translate_image(defect_mask, x_offset, y_offset, result_image.shape[1], result_image.shape[0])

        # Rotation 적용
        if operation == 'rotate' or operation == 'both':
            angle = random.randint(params['min_rotation'], params['max_rotation'])
            current_defect_image, current_defect_mask = rotate_image(current_defect_image, current_defect_mask, angle, result_image.shape[1], result_image.shape[0])

        # 경계를 벗어나는지 체크하고 조정
        if not is_within_boundary(current_defect_mask, boundary_mask):
            # 경계 내에 있는 영역만 사용하거나 조정
            continue  # 또는 적절한 조정 로직 적용

        # 결함 이미지와 마스크 적용
        alpha_mask = current_defect_mask / 255.0
        for c in range(3):
            result_image[:, :, c] = current_defect_image[:, :, c] * alpha_mask + result_image[:, :, c] * (1 - alpha_mask)

        result_images.append(result_image)
        result_masks.append(current_defect_mask)  # 변형된 마스크 저장
 
    return result_images, result_masks  # 이미지와 마스크 모두 반환

def translate_image(image, x_offset, y_offset, max_width, max_height):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def rotate_image(image, mask, angle, max_width, max_height):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    return rotated_image, rotated_mask

# Translation 예시
# translated_image = paste_defect_on_normal_with_constraints(
#     normal_image, defect_image, defect_mask, boundary_mask,
#     'translate', {'x_offset': 100, 'y_offset': 100},
#     num_augmentations=1  # 생성하고 싶은 증강 이미지의 갯수
# )

# Rotation 예시 (특정 각도로 회전만 원할 경우)
# rotated_image = paste_defect_on_normal_with_constraints(
#     normal_image, defect_image, defect_mask, boundary_mask,
#     'rotate', {'angle': 45},
#     num_augmentations=1  # 생성하고 싶은 증강 이미지의 갯수
# )

# 데이터 증강 실행 예시 및 파라미터 설정
image_dir = ''  # 이미지 폴더 경로 설정
normal_image_path = os.path.join(image_dir, "000_normal.png")
boundary_mask_path = os.path.join(image_dir, "000_mask_1.bmp")
defect_image_path = os.path.join(image_dir, "000_cut.png")
defect_mask_path = os.path.join(image_dir, "000_mask_cut.png")

normal_image = cv2.imread(normal_image_path)
defect_image = cv2.imread(defect_image_path)
defect_mask = cv2.imread(defect_mask_path, cv2.IMREAD_GRAYSCALE)
boundary_mask = cv2.imread(boundary_mask_path, cv2.IMREAD_GRAYSCALE)

augmentation_params = {
    'min_x_offset': -100, 'max_x_offset': 100,
    'min_y_offset': -100, 'max_y_offset': 100,
    'min_rotation': -45, 'max_rotation': 45
}

# augmented_images = paste_defect_on_normal_with_constraints(
#     normal_image, defect_image, defect_mask, boundary_mask,
#     'both', augmentation_params, num_augmentations=10
# )

# 데이터 증강 및 결과 저장
augmented_images, augmented_masks = paste_defect_on_normal_with_constraints(
    normal_image, defect_image, defect_mask, boundary_mask,
    'both', augmentation_params, num_augmentations=100
)

results_path = create_results_directory()

for i, (img, mask) in enumerate(zip(augmented_images, augmented_masks)):
    cv2.imwrite(os.path.join(results_path, f'augmented_image_{i}.png'), img)
    # 마스크는 그레이스케일 이미지로 저장
    cv2.imwrite(os.path.join(results_path, f'augmented_mask_{i}.png'), mask)

