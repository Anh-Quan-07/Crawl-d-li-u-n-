import cv2
import numpy as np
import os
# Nạp dữ liệu
data = []
labels = []

for filename in os.listdir("D:\Downloads\các tài liệu kì 1 này\Môn thị giác máy tính\ảnh gấu trúc"):
    img = cv2.imread(os.path.join("D:\Downloads\các tài liệu kì 1 này\Môn thị giác máy tính\ảnh gấu trúc", filename))
    data.append(img.reshape(-1))
    labels.append(filename.split("_")[0])

# Trích xuất đặc trưng
features = np.zeros((len(data), 256))

for i in range(len(data)):
    img = data[i]
    features[i] = cv2.calcHist([img], [0], None, [256], [0, 255])

# Huấn luyện mô hình
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(features, labels)

# Phân loại hình ảnh
img = cv2.imread("6_9.png")
features = cv2.calcHist([img], [0], None, [256], [0, 255])
predictions = clf.predict(features)

# In kết quả
print("Hình ảnh được phân loại là {}".format(labels[predictions[0]]))