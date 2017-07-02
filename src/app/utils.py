import os

# путь до исходных данных
DATASET_PATH = "./dataset/"


def create_dataset():
    from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

    """
    функция, вызываемая извне (из главной программы) для генерации набора данных
    :return:
    """

    def logging():
        """
        Отчет о созданных данных
        :return: None
        """
        print("\nИз исходного набора данных в папках '/dataset/markup' и '/dataset/unmarkup'")
        print("создан и сохранен в папку '/dataset/preview/' расширенный набор данных,")
        print("полученный из исходных посредством применения следующих преобразований:")
        print("\t* вращение")
        print("\t* горизонтальное и вертикальное смещение")
        print("\t* приближение")
        print("\t* размытие")
        cnt = len([name for name in os.listdir('./dataset/preview/markup/')
                   if os.path.isfile('./dataset/preview/markup/' + name)]) + \
              len([name for name in os.listdir('./dataset/preview/unmarkup/')
                   if os.path.isfile('./dataset/preview/unmarkup/' + name)])
        print("Итого: " + str(cnt) + " файлов создано.")

    def generate_dataset(clsfy, data_generator):
        """
        функция генерирует большой датасет на основе ограниченного числа фотографий
        :param clsfy: строка
            содержащая путь к файлам и семантический смысл генерируемых объектов
            так, в нашем случае, если это
            'markup/', то это папка с фотографиями дорожной разметки пешеходных переходов, а если
            'unmarkup/', то фото дороги без разметки пешеходного

        :param data_generator: объект-генератор тестового набора данных
        :return: None
        """
        dataset_path = DATASET_PATH + clsfy  # путь до фотографий
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
            # _ означает batch, но не используется
            for _ in data_generator.flow(x, batch_size=1,
                                         save_to_dir=DATASET_PATH + 'preview/' + clsfy,
                                         save_prefix=clsfy[0:-1],
                                         save_format='jpeg'):
                i += 1
                if i > 20:
                    break

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    generate_dataset("markup/", data_generator=datagen)
    generate_dataset("unmarkup/", data_generator=datagen)
    logging()
