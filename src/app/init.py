from skimage import exposure
from scipy import ndimage as ndi
from skimage.io import imread, imsave
from skimage.filters import roberts
from skimage.transform import downscale_local_mean
import os.path


OUTPUT_PREFIX = './output/'
DATA_PATH = './dataset/'


def make_model_from_dataset():
    """
    Основная функция модуля, вызывается извне.
    Для каждой картинки папки ./dataset применяет необходимые фильтры
    для получения изображений в чб, с выделенными контурами объектов
    :return: None
    """

    def logging(folder, count, is_total):
        """
        выфв
        :param folder:
        :param count:
        :param is_total:
        :return:
        """
        print("\nПо итогу сформировано " + str(count) + " файлов в каталоге ./output/" + folder)
        print("К изображением применены:")
        print("\t* черно-белый фильтр")
        print("\t* размытие гаусса фильтр")
        print("\t* выделение краев Робертса")
        if is_total:
            print("файлы разложены по соответствующим папкам markup и unmarkup")
            print("Модуль завершил работу")

    def make_edge_image(res, out_name, folder):
        """
        функция применяет алгоритм roberts к чернобелому изображению
        для выделения контуров объектов и сохраняет результат в файл
        :param res: изображение, к которому надо применить алгоритм
        :param out_name: имя файла, в который произведем сохранение
        :param folder: папка
        :return: None
        """
        img_roberts = roberts(res)
        img_roberts = downscale_local_mean(img_roberts, (2, 2))
        filename = OUTPUT_PREFIX + folder + out_name + '.jpg'
        imsave(filename, img_roberts)

    def get_gray_scale(filename):
        """
        Функция создает чернобелое изображение с применением размытия по Гауссу
        из исходного.
        Полученное изображение необходимо для выделения контуров объекта
        :param filename: имя исходного изображение
        :return: чернобелое изображение с размытием
        """
        rgb = imread(filename)  # [200: 400, 300:800]
        src = rgb[:, :, 2]  # получили синюю компоненту RGB
        sig = exposure.adjust_sigmoid(src)
        gauss = ndi.gaussian_filter(sig, 3)  # размытие по Гауссу

        return exposure.adjust_sigmoid(gauss)

    def collect_images(folder):
        """
        Агрегирование всех обработанных фотографий из папки
        DATA_PATH + folder
        в выходную папку с помощью вызова функции make_edge_image
        для каждого файла папки
        :param folder: папка, откуда брать данных
        :return: None
        """
        i = 0
        data_path = DATA_PATH + folder

        the_list = [name for name in os.listdir(data_path)
                    if os.path.isfile(os.path.join(data_path, name))]

        for name in the_list:
            result = get_gray_scale(data_path + name)
            make_edge_image(result, folder[0:-1] + '-' + str(i), folder)
            i += 1
        return i

    cnt1 = collect_images("markup/")
    cnt2 = collect_images("unmarkup/")
    logging("markup/", cnt1, False)
    logging("unmarkup/", cnt2, False)
    logging("", cnt1 + cnt2, True)
