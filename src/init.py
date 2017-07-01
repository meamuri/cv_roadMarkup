from skimage import exposure
from scipy import ndimage as ndi

from skimage.io import imread, imsave
from skimage.filters import roberts

OUTPUT_PREFIX = 'output/'


def make_edge_image(res, out_name):
    img_roberts = roberts(res)
    # img_roberts = ndi.gaussian_filter(img_roberts, 3)
    filename = OUTPUT_PREFIX + out_name + '-roberts.jpg'
    imsave(filename, img_roberts)


def get_gray_scale(filename):
    """
    Функция создает чернобелое изображение с применением размытия по Гауссу
    из исходного.
    Полученное изображение необходимо для выделения контуров объекта
    :param filename: имя исходного изображение
    :return: чернобелое изображение с размытием
    """
    rgb = imread(filename)

    src = rgb[:, :, 2]  # получили синюю компоненту RGB
    # out_name = OUTPUT_PREFIX + 'src.jpg'
    # imsave(out_name, src)

    sig = exposure.adjust_sigmoid(src)
    # test(sig, "sig")

    gauss = ndi.gaussian_filter(sig, 3)  # размытие по Гауссу
    # test(gauss, "gauss1")

    return exposure.adjust_sigmoid(gauss)


def make_model_from_dataset():
    import os.path
    dataset_path = "./dataset"
    cnt = len([name for name in os.listdir(dataset_path)
               if os.path.isfile(os.path.join(dataset_path, name))])
    for i in range(1, cnt + 1):
        prefix = str(i)
        filename = 'dataset/'+ prefix + '.jpg'
        result = get_gray_scale(filename)
        make_edge_image(result, "result" + prefix)
