from __future__ import division
from __future__ import print_function

from sklearn import linear_model
from skimage import data
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics

from io import StringIO
from PIL import Image
import os


def process_directory(directory):
    """Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    """
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    """
    Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    """
    image_fp = StringIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None


def process_image(the_image, blocks=4):
    """Given a PIL Image object it returns its feature vector.

    Args:
      the_image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    """
    if not the_image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    pixel_count = 0
    for pixel in the_image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]


def get_images(path):
    path = './output/' + path
    names = [path + name for name in os.listdir(path)
             if os.path.isfile(path + name)]
    arrays = [data.imread(name) for name in names]

    return arrays


def get_coefs(arrays):
    reg = linear_model.LinearRegression()

    reg.fit(X=arrays)

    return reg.coef_


def bin_objects(X, y, ):
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


def train(training_path_a='markup/', training_path_b='unmarkup/', print_metrics=True):
    """
    Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    """
    training_a = process_directory('output/' + training_path_a)  # get_images(training_path_a)
    training_b = get_images('output/' + training_path_b)
    # data contains all the training data (a list of feature vectors)
    the_data = training_a + training_b
    # target is the list of target classes for each feature vector: a '1' for
    # class A and '0' for class B
    target = [1] * len(training_a) + [0] * len(training_b)
    # split training data in a train set and a test set. The test set will
    # contain 20% of the total
    x_train, x_test, y_train, y_test = model_selection.train_test_split(the_data,
                                                                        target, test_size=0.20)
    # define the parameter search space
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = model_selection.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
                                            classifier.predict(x_test)))
    return classifier
