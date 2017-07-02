from __future__ import division
from __future__ import print_function

from sklearn import linear_model
from skimage import io
import os


def make_network():
    return linear_model.LinearRegression()


def get_images():
    # import numpy as np
    path = './output/markup'
    m_up_len = len([path + name for name in os.listdir(path)
                    if os.path.isfile(path + name)])
    path = './output/unmarkup'
    un_m_up_len = len([path + name for name in os.listdir(path)
                       if os.path.isfile(path + name)])
    obj = {
        'arrays': io.imread_collection('./output/markup/*.jpg:./output/markup/*.jpg').concatenate(),
        'len_mark': m_up_len,
        'len_unmark': un_m_up_len,
    }
    return obj


def get_reg():
    reg = linear_model.LinearRegression()
    return reg

    # obj = get_images()
    # X = obj['arrays']
    #
    # y = [1] * obj['len_mark'] + [0] * obj['len_unmark']
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(X,
    #
    # reg.fit(X=x_train, y=y_train)


# def smthng_that_doesnt_work(training_path_a='markup/', training_path_b='unmarkup/', print_metrics=True):
#     """
#     Trains a classifier. training_path_a and training_path_b should be
#     directory paths and each of them should not be a subdirectory of the other
#     one. training_path_a and training_path_b are processed by
#     process_directory().
#
#     Args:
#       training_path_a (str): directory containing sample images of class A.
#       training_path_b (str): directory containing sample images of class B.
#       print_metrics  (boolean, optional): if True, print statistics about
#         classifier performance.
#
#     Returns:
#       A classifier (sklearn.svm.SVC).
#     """
#     training_a = process_directory('output/' + training_path_a)
#     training_b = process_directory('output/' + training_path_b)
#     # training_a = get_images(training_path_a)
#     # training_b = get_images(training_path_b)
#
#     # data contains all the training data (a list of feature vectors)
#     the_data = training_a + training_b
#     # target is the list of target classes for each feature vector: a '1' for
#     # class A and '0' for class B
#     target = [1] * len(training_a) + [0] * len(training_b)
#     # split training data in a train set and a test set. The test set will
#     # contain 20% of the total
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(the_data,
#                                                                         target, test_size=0.20)
#     # define the parameter search space
#     parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
#                   'gamma': [0.01, 0.001, 0.0001]}
#     # search for the best classifier within the search space and return it
#     clf = model_selection.GridSearchCV(svm.SVC(C=1), parameters).fit(x_train, y_train)
#     classifier = clf.best_estimator_
#     if print_metrics:
#         print()
#         print('Parameters:', clf.best_params_)
#         print()
#         print('Best classifier score')
#         print(metrics.classification_report(y_test,
#                                             classifier.predict(x_test)))
#     return classifier
