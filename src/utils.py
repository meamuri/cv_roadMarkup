from skimage.io import imread, imsave
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny


def test(res, out_name):
    # filename = out_name + '.jpg'
    # imsave(filename, res)

    img_roberts = roberts(res)
    filename = out_name + '-roberts.jpg'
    imsave(filename, img_roberts)

    img_sobel = sobel(res)
    filename = out_name + '-sobel.jpg'
    imsave(filename, img_sobel)
    #
    # img_canny = canny(res)
    # filename = out_name + '-canny.jpg'
    # imsave(filename, img_canny)
    #
    # img_canny3 = canny(res, sigma=0.5)
    # filename = out_name + '-canny3.jpg'
    # imsave(filename, img_canny3)
    #
    # img_scharr = scharr(res)
    # filename = out_name + '-scharr.jpg'
    # imsave(filename, img_scharr)

    img_prewitt = prewitt(res)
    filename = out_name + '-prewitt.jpg'
    imsave(filename, img_prewitt)

