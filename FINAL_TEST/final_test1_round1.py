import numpy as np
import os

from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score


def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            img_new = io.imread(os.path.join(img_file))
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
    cat_path = 'C:/Marchine_learning/Preprocessing_image/PetImages/Cat'
    dog_path = 'C:/Marchine_learning/Preprocessing_image/PetImages/Dog'
    images, labels = read_img_data(cat_path, 'cat', (32, 32))
    img_dog, label_dog = read_img_data(dog_path, 'dog', (32, 32))
    images.extend(img_dog)
    labels.extend(label_dog)
    X = np.array(images)
    y = LabelBinarizer().fit_transform(labels)
    return X, y

# Hàm chia train-test và chuẩn hóa dữ liệu
def featureScalingSplit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=15)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    return X_train, X_test, y_train, y_test

# Mô hình Logistic RegressionCV
def logistic_regression_cv(X_train, X_test, y_train):
    lgt = LogisticRegression(solver='liblinear', max_iter=1500, multi_class='auto')
    lgt.fit(X_train, y_train)
    y_pred = lgt.predict(X_test)
    print('Kết quả huấn luyên 10-fold cv của mô hình LogisticRegression:')
    scores = cross_val_score(lgt, X_train, y_train, cv=10, scoring='accuracy')
    print(scores)
    return lgt, scores, y_pred

# Mô hình k-NN
def kNN_cv(X_train, X_test, y_train):
    knn = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='uniform')
    knn_score = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('Kết quả huấn luyên 10-fold cv của mô hình k-NN:')
    print(knn_score)
    return knn_score, y_pred, knn

# Hàm đánh giá hiệu năng
def evaluate_Model(y_test, y_pred):
    print("Accuracy: {}%".format(round(accuracy_score(y_test, y_pred), 5)))
    print("Precision score: {}%".format(round(precision_score(y_test, y_pred), 5)))
    print("Recall score {}%".format(round(recall_score(y_test, y_pred), 5)))
    print("F1 Score {}%".format(round(f1_score(y_test, y_pred, average='weighted'), 5)))
    print("Haming loss: ", hamming_loss(y_test, y_pred))

import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    print("hihihihihihi")
    # Bước 1: Đọc dữ liệu
    X, y = build_imd_data()
    # Bước 2: Phân chia train - test, chuẩn hóa dữ liệu
    X_train, X_test, y_train, y_test = featureScalingSplit(X, y)
    # Kết quả mô hình logistic regression
    scores = logistic_regression_cv()
    print(scores)
    # Kết quả mô hình k-NN
    knn_score = kNN_cv()
    print(knn_score)
    # Confusion matrix
    y_pred = logistic_regression_cv()
    confusion = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion, classes=['y pred', 'y test'], title='Confusion matrix')

if __name__ == '__main__':
    main()