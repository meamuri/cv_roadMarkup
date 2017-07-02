from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.feature import local_binary_pattern
from skimage.transform import rotate

# settings for LBP
RADIUS = 2
N_POINTS = 8 * RADIUS
METHOD = 'uniform'


def show_graphics():
    """
    Функция позволяет продемонстрировать корреляцию гистограмм изображений
    с разметкой и без нее.
    При этом берется случайная фотография из набора размеченных и случайная из фото без разметки
    :return: None
    """
    def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                       facecolor='0.5')

    plt.rcParams['font.size'] = 9

    def divergence(p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        filt = np.logical_and(p != 0, q != 0)
        return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

    def match(refs, img):
        best_score = 10
        best_name = None
        lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        for name, ref in refs.items():
            ref_hist, _ = np.histogram(ref, normed=True, bins=n_bins,
                                       range=(0, n_bins))
            score = divergence(hist, ref_hist)
            if score < best_score:
                best_score = score
                best_name = name
        return best_name

    def read_rand_pic_from_folder(folder):
        """
        выбирает случайную фотографию из предложенной папки 'markup/' или 'unmarkup/'
        :param folder:
        :return:
        """
        import random
        import os
        path = './output/' + folder
        cnt = len([name for name in os.listdir(path)
                   if os.path.isfile(os.path.join(path, name))])
        dig = random.randint(0, cnt - 1)
        path += folder[0:-1] + '-' + str(dig) + '.jpg'
        return data.imread(path)

    markup = read_rand_pic_from_folder('markup/')
    unmarkup = read_rand_pic_from_folder('unmarkup/')

    refs = {
        'markup': local_binary_pattern(markup, N_POINTS, RADIUS, METHOD),
        'unmarkup': local_binary_pattern(unmarkup, N_POINTS, RADIUS, METHOD),
    }

    # classify rotated textures
    print('Rotated images matched against references using LBP:')
    print('original: markup, rotated: 30deg, match result: ',
          match(refs, rotate(markup, angle=30, resize=False)))
    print('original: unmarkup, rotated: 70deg, match result: ',
          match(refs, rotate(unmarkup, angle=70, resize=False)))

    # plot histograms of LBP of textures
    fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(9, 6))
    plt.gray()

    ax1.imshow(markup)
    ax1.axis('off')
    hist(ax4, refs['markup'])
    ax4.set_ylabel('Percentage')

    ax2.imshow(unmarkup)
    ax2.axis('off')
    hist(ax5, refs['unmarkup'])
    ax5.set_xlabel('Uniform LBP values')

    plt.show()
