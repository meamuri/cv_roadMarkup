from skimage import exposure
from scipy import ndimage as ndi
from skimage.filters import roberts


def process_image(rgb):
    src = rgb[:, :, 2]  # получили синюю компоненту RGB
    sig = exposure.adjust_sigmoid(src)
    gauss = ndi.gaussian_filter(sig, 3)  # размытие по Гауссу
    res = exposure.adjust_sigmoid(gauss)

    img_roberts = roberts(res)

    return img_roberts
