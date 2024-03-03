import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

def calculate_mask_area(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    area = cv2.countNonZero(mask)
    return area

# 수정된 이미지와 마스크가 저장된 폴더 경로
image_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/test/crack'
mask_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/ground_truth/crack'

# 이미지와 마스크 파일 목록 불러오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

# 마스크 면적 계산
mask_areas = np.array([calculate_mask_area(os.path.join(mask_dir, mask)) for mask in tqdm(mask_files, desc='Calculating Mask Areas')])

# 면적을 기반으로 클러스터링
scaler = StandardScaler()
scaled_areas = scaler.fit_transform(mask_areas.reshape(-1, 1))  # 면적 데이터 스케일링

n_clusters = 5  # 클러스터 수 설정
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(scaled_areas)
labels = kmeans.labels_

# 클러스터링 결과에 따라 폴더 생성 및 이미지와 마스크 복사
output_dir = '/Users/kimgwangyeol/Desktop/AiV/biased_area/crack' 

# 클러스터링 결과에 따라 폴더 생성 및 이미지와 마스크 복사
for i in range(n_clusters):
    cluster_dir = os.path.join(output_dir, f'cluster_{i}')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # 이미지 파일과 마스크 파일을 올바르게 복사
    for img_path, mask_path, label in tqdm(zip(image_files, mask_files, labels), total=len(image_files), desc=f'Copying Images and Masks to cluster_{i}'):
        if label == i:
            # 이미지 파일 복사
            shutil.copy(os.path.join(image_dir, img_path), cluster_dir)  
            # 마스크 파일 복사: 마스크 파일이 'ground_truth' 폴더에 있으므로, 해당 경로에서 복사
            shutil.copy(os.path.join(mask_dir, mask_path), cluster_dir)

# 클러스터링 결과 시각화
plt.scatter(np.arange(len(mask_areas)), mask_areas, c=labels, cmap='viridis')
plt.xlabel('Mask Index')
plt.ylabel('Scaled Mask Area')
plt.title('Clustering Results Based on Mask Area')
plt.show()
