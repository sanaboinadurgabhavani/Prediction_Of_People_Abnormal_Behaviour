import pandas as pd  # for data manipulation
from django.conf import settings
from pathlib import Path

import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
import skimage


def load_image_files(container_path, dimension=(104, 104,3)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "Prediction of Abnormal Human Behaviour Using Machine Learning"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)

    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


path = settings.MEDIA_ROOT + "//" + "train"
image_dataset = load_image_files(path)  # Load here dataset
print(image_dataset.target_names)

X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3,
                                                    random_state=109)
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]


def process_randomForest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # import pickle
    # # now you can save it to a file
    # with open(r'abnormal_detection.alex', 'wb') as f:
    #     pickle.dump(clf, f)
    print(clf.score(X_test, y_test))
    rf_report = metrics.classification_report(y_test, y_pred,output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, rf_report))
    return rf_report


def process_decesionTree():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    dt_report = metrics.classification_report(y_test, y_pred,output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, dt_report))
    return dt_report


def process_knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    knn_report = metrics.classification_report(y_test, y_pred,output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, knn_report))
    return knn_report


def process_ksvm():
    from sklearn.svm import SVC
    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    ksvm_report = metrics.classification_report(y_test, y_pred,output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, ksvm_report))
    return ksvm_report


def process_lsvm():
    from sklearn.svm import SVC
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    lsvm_report = metrics.classification_report(y_test, y_pred,output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, lsvm_report))
    return lsvm_report
