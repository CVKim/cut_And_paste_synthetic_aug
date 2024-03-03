import os
import re
import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

def contains_korean(text):
    return bool(re.search('[\uac00-\ud7a3]', text))

def extract_features_vgg16(img_path):
    # VGG16 모델을 여러 번 로드하지 않도록 수정
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    return flattened_features

# VGG16 모델 로드 및 모델 설정
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

image_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/test/crack'
mask_dir = '/Volumes/T7 SSD/Dataset/mvtec_anomaly_detection/hazelnut/ground_truth/crack'

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and not contains_korean(f)])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') and not contains_korean(f)])

paired_files = [(os.path.join(image_dir, img), os.path.join(mask_dir, msk)) 
                for img, msk in tqdm(zip(image_files, mask_files), total=len(image_files), desc='Pairing Images and Masks') if img[:3] == msk[:3]]

# 이미지 및 마스크 특성 추출
features = []
for img_path, mask_path in tqdm(paired_files, desc='Extracting Features'):
    # VGG16 특성 추출
    dl_features = extract_features_vgg16(img_path)
    
    # 마스크 기하학적 특성 추출
    mask = cv2.imread(mask_path, 0)
    moments = cv2.moments(mask)
    cx = int(moments['m10']/moments['m00']) if moments['m00'] != 0 else 0
    cy = int(moments['m01']/moments['m00']) if moments['m00'] != 0 else 0
    area = cv2.countNonZero(mask)
    geometric_features = np.array([cx, cy, np.log1p(area)])
    
    # 특성 결합
    combined_features = np.hstack((dl_features, geometric_features))
    features.append(combined_features)

features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)  # 특성 스케일링

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(features_scaled)
labels = kmeans.labels_

output_dir = '/Users/kimgwangyeol/Desktop/AiV/biased_test/crack'
for i in range(n_clusters):
    cluster_dir = os.path.join(output_dir, f'cluster_{i}')
    os.makedirs(cluster_dir, exist_ok=True)
    
    for (image_path, mask_path), label in tqdm(zip(paired_files, labels), total=len(paired_files), desc=f'Copying Images and Masks to cluster_{i}'):
        if label == i:
            shutil.copy(image_path, cluster_dir)
            mask_file_name = os.path.basename(mask_path)
            shutil.copy(mask_path, os.path.join(cluster_dir, mask_file_name))

plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Results with Combined Features')
graph_path = os.path.join(output_dir, 'clustering_results_with_combined_features.png')
plt.savefig(graph_path)
plt.show()
