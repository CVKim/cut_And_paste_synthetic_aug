import cv2
import numpy as np
import random

def paste_defect_on_normal_randomly(defect_image, defect_mask, normal_image, boundary_mask):
    
    result_image = normal_image.copy()

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # 경계 마스크에서 결함을 배치할 수 있는 위치 찾기
    boundary_coords = np.where(boundary_mask == 255)
    if boundary_coords[0].size == 0 or boundary_coords[1].size == 0:
        raise ValueError("No valid area found in boundary mask")

    while True:
        upper_left_x = random.choice(boundary_coords[1])
        upper_left_y = random.choice(boundary_coords[0])

        if upper_left_x + w >= result_image.shape[1] or upper_left_y + h >= result_image.shape[0]:
            # 랜덤 위치가 이미지 경계를 넘어가면 다시 시도
            continue

        # 선택된 위치에 결함이 배치 가능한지 확인
        if np.all(boundary_mask[upper_left_y:upper_left_y + h, upper_left_x:upper_left_x + w] == 255):
            break

    defect_rgba = cv2.cvtColor(defect_image, cv2.COLOR_RGB2RGBA)
    alpha_channel = np.zeros(defect_image.shape[:2], dtype=defect_image.dtype)

    cv2.drawContours(alpha_channel, [contour], -1, (255), thickness=cv2.FILLED)
    defect_rgba[:, :, 3] = alpha_channel

    defect_part = defect_rgba[y:y+h, x:x+w]

    for c in range(0, 3):
        result_image[upper_left_y:upper_left_y+h, upper_left_x:upper_left_x+w, c] = \
            defect_part[:, :, c] * (defect_part[:, :, 3] / 255.0) + \
            result_image[upper_left_y:upper_left_y+h, upper_left_x:upper_left_x+w, c] * \
            (1.0 - defect_part[:, :, 3] / 255.0)

    return result_image

def paste_defect_on_normal_with_transparency(defect_image, defect_mask, normal_image):
    
    result_image = normal_image.copy()
    
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    defect_rgba = cv2.cvtColor(defect_image, cv2.COLOR_RGB2RGBA)
    alpha_channel = np.zeros(defect_image.shape[:2], dtype=defect_image.dtype)

    cv2.drawContours(alpha_channel, [contour], -1, (255), thickness=cv2.FILLED)
    defect_rgba[:, :, 3] = alpha_channel

    defect_part = defect_rgba[y:y+h, x:x+w]

    # 타겟 이미지에 붙이기 전 알파 채널을 이용해 배경 처리
    for c in range(0, 3):
        result_image[y:y+h, x:x+w, c] = defect_part[:, :, c] * (defect_part[:, :, 3] / 255.0) + \
                                        result_image[y:y+h, x:x+w, c] * (1.0 - defect_part[:, :, 3] / 255.0)

    return result_image

def rotate_and_paste_defect(defect_image, defect_mask, normal_image, x_offset, y_offset, angle):
    result_image = normal_image.copy()

    # Find contours in the defect mask
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the center for rotation
    center = (x + w // 2, y + h // 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the defect image and mask
    rotated_defect_image = cv2.warpAffine(defect_image, M, (normal_image.shape[1], normal_image.shape[0]))
    rotated_defect_mask = cv2.warpAffine(defect_mask, M, (normal_image.shape[1], normal_image.shape[0]))

    # Calculate new bounding box for the rotated mask
    rotated_contours, _ = cv2.findContours(rotated_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not rotated_contours:
        raise ValueError("No contours found in rotated mask")
    rotated_contour = max(rotated_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(rotated_contour)

    # Apply the offsets
    target_x = max(0, min(normal_image.shape[1] - rw, x + x_offset))
    target_y = max(0, min(normal_image.shape[0] - rh, y + y_offset))

    # Create a mask for blending
    alpha_mask = rotated_defect_mask[ry:ry+rh, rx:rx+rw] / 255.0
    alpha_mask_3d = np.dstack([alpha_mask, alpha_mask, alpha_mask])

    # Blend the rotated defect into the target image
    for c in range(3):
        result_image[target_y:target_y+rh, target_x:target_x+rw, c] = \
            alpha_mask_3d[:, :, c] * rotated_defect_image[ry:ry+rh, rx:rx+rw, c] + \
            (1 - alpha_mask_3d[:, :, c]) * result_image[target_y:target_y+rh, target_x:target_x+rw, c]

    return result_image

defect_image = cv2.imread("000_Defect.png", cv2.IMREAD_COLOR)
defect_mask = cv2.imread("000_mask.png", cv2.IMREAD_GRAYSCALE)

normal_image = cv2.imread("000_normal.png", cv2.IMREAD_COLOR)
noraml_boundary = cv2.imread("000_mask_1.bmp", cv2.IMREAD_GRAYSCALE)

result_image_random = paste_defect_on_normal_randomly(defect_image, defect_mask, normal_image, noraml_boundary)
cv2.imwrite("result_image_random.png", result_image_random)

result_image = paste_defect_on_normal_with_transparency(defect_image, defect_mask, normal_image)
cv2.imwrite("result_image.png", result_image)

# x, y, angle offset 정의
x_offset = 2
y_offset = 5
angle = 50

result_image_offset = rotate_and_paste_defect(defect_image, defect_mask, normal_image, x_offset, y_offset, angle)
cv2.imwrite("result_image_offset.png", result_image_offset)