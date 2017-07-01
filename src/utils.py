from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


DATASET_PATH = "./dataset/"


def generate_dataset(clsfy, datagen):
    """

    :param clsfy: "markup/" or "unmarkup/"
    :param datagen:
    :return:
    """
    dataset_path = DATASET_PATH + clsfy
    im_list = [name for name in os.listdir(dataset_path)
               if os.path.isfile(os.path.join(dataset_path, name))]
    for ind, name in enumerate(im_list):
        filename = dataset_path + name
        img = load_img(filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=DATASET_PATH + 'preview/' + clsfy,
                                  save_prefix=clsfy[0:-1],
                                  save_format='jpeg'):
            i += 1
            if i > 20:
                break


