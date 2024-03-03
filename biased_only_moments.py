import os
import re
import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

def contains_korean(text):
    return bool(re.search('[\uac00-\ud7a3]', text))

# 수정된 이미지와 마스크가 저장된 폴더 경로
image_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/test/crack'
mask_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/ground_truth/crack'

# 한글로 되어 있는 이미지 파일을 제외하고, 파일명 길이가 4자 이하인 이미지와 마스크 파일 목록 불러오기
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and not contains_korean(f) and len(os.path.splitext(f)[0]) <= 4])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') and not contains_korean(f)])

# 파일명 앞 3자리 기준으로 이미지와 마스크 짝지어주기 과정에 tqdm 추가
paired_files = [(os.path.join(image_dir, img), os.path.join(mask_dir, msk)) 
                for img, msk in tqdm(zip(image_files, mask_files), total=len(image_files), desc='Pairing Images and Masks') if img[:3] == msk[:3]]

def extract_center_features(mask_path):
    mask = cv2.imread(mask_path, 0)
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return np.array([0, 0])
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    return np.array([cx, cy])

# 중심 좌표 기반 특성 벡터 추출
features = np.array([extract_center_features(mask) for _, mask in tqdm(paired_files, desc='Extracting Center Features')])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # 특성 스케일링

# KMeans 클러스터링
n_clusters = 5  # 클러스터 수 설정
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(features_scaled)
labels = kmeans.labels_

# 클러스터링 결과에 따라 폴더 생성 및 이미지와 마스크 복사
output_dir = '/Users/kimgwangyeol/Desktop/AiV/biased_only_moments/crack' 
for i in range(n_clusters):
    cluster_dir = os.path.join(output_dir, f'cluster_{i}')
    os.makedirs(cluster_dir, exist_ok=True)
    
    for (image_path, mask_path), label in tqdm(zip(paired_files, labels), total=len(paired_files), desc=f'Copying Images and Masks to cluster_{i}'):
        if label == i:
            shutil.copy(image_path, cluster_dir)  # 이미지 복사
            mask_file_name = os.path.basename(mask_path)
            shutil.copy(mask_path, os.path.join(cluster_dir, mask_file_name))  # 마스크 복사

# 클러스터링 결과 그래프로 시각화
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis')
plt.xlabel('Scaled Center X')
plt.ylabel('Scaled Center Y')
plt.title('Clustering Results Based on Center Coordinates')

graph_path = os.path.join(output_dir, 'clustering_results_center_coordinates.png')
plt.savefig(graph_path)
plt.show()
