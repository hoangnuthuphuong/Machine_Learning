import numpy as np
import os
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Hàm kiểm tra ảnh có lỗi hay không
def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            io.imread(os.path.join(img_file))
        return False
    except Exception as e:
        print(e)
        return True

# Hàm chuyển ảnh màu thành vector 1024
# Đọc ảnh -> xám, resize ,làm phẳng
def read_img_data(path,label, size):
    labels = []
    img_data = []
    # xác định đường dẫn và biến dữ  ảnh
    files = os.listdir(path)
    # đọc tất cả file trong path
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray=True)
            img = resize(img, size)
            img_vector = img.flatten()

            labels.append(labels)
            img_data.append(img_vector)
    labels = np.array(labels)
    return labels, img_data

# Hàm xây dựng CSDL ảnh
def build_imd_data():
    cat_path = 'C:/Users/DELL/PycharmProjects/HNTP/PetImages/Cat'
    dog_path = 'C:/Users/DELL/PycharmProjects/HNTP/PetImages/Dog'
    images, labels = read_img_data(cat_path,'cat', (32, 32))
    img_dog, label_dog = read_img_data(dog_path, 'dog', (32, 32))
    images.extend(img_dog)
    labels.extend(label_dog)
    X = np.array(images)
    y = LabelBinarizer().fit_transform(labels)
    return X, y

def featureScalingSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=15)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, X_test, y_train, y_test

# Số lượng k-fold được xác định tùy thuộc vào số lượng y_train
def checkkFold(y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    result = dict(zip(unique, counts))
    print("Số lượng k-fold được xác định")
    print(result)
    return result


def crossValScore(model, X_train, y_train, cv=10, scoring='accuracy'):
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
    print('\nKết quả huấn luyện 10-fold cv:')
    print(scores)
    return scores



def main():
    print("hihihihihihi")
    # Bước 1: Đọc dữ liệu
    X, y = build_imd_data()

    #print("x shape", X.shape)

    #Bước 2: Phân chia train - test theo tỉ lệ 70% - 30%
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)

    # Bước 3: Chuẩn hóa dữ liệu
    X_train, X_test = featureScalingSplit(X_train, X_test)

    #Số lượng k-fold được xác định tùy thuộc vào số lượng y_train
    result = checkkFold(y_train)

    #Bước 4: Khởi tạo mô hình hồi quy logistic, với thuật toán tối ưu là liblinear
    #Bước lặp 1500; multi_class = 'auto' để tự phát hiện nhãn lớp nhị phân hay đa nhãn lớp
    classifier = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')

    #Bước 5: Huấn luyện mô hình cv = 10 và độ đo là scoring='accuracy' và in kết quả
    scores = crossValScore(classifier, X_train, y_train, cv=10, scoring='accuracy')

if __name__ == '__main__':
    main()