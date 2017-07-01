from skimage import exposure
from scipy import ndimage as ndi

from skimage.io import imread, imsave
from skimage.filters import roberts, sobel, scharr, prewitt


def test(res, out_name):
    img_roberts = roberts(res)
    # img_roberts = ndi.gaussian_filter(img_roberts, 3)
    filename = 'output/' + out_name + '-roberts.jpg'
    imsave(filename, img_roberts)

OUTPUT_PREFIX = 'output/'


def main():
    filename = 'dataset/3.jpg'
    rgb = imread(filename)

    src = rgb[:, :, 2]
    out_name = OUTPUT_PREFIX + 'src.jpg'
    imsave(out_name, src)

    sig = exposure.adjust_sigmoid(src)
    # test(sig, "sig")

    gauss = ndi.gaussian_filter(sig, 3)
    # test(gauss, "gauss1")

    result = exposure.adjust_sigmoid(gauss)
    test(result, "result")

if __name__ == "__main__":
    main()