from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
import os

# settings for LBP
RADIUS = 2
N_POINTS = 8 * RADIUS


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


METHOD = 'uniform'
plt.rcParams['font.size'] = 9


def kullback_leibler_divergence(p, q):
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
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def just_do_it():
    markup = data.imread('./output/markup/result-1.jpg')
    unmarkup = data.imread('./output/unmarkup/result-1.jpg')
    # wall = data.load('rough-wall.png')

    refs = {
        'brick': local_binary_pattern(markup, N_POINTS, RADIUS, METHOD),
        'grass': local_binary_pattern(unmarkup, N_POINTS, RADIUS, METHOD),
        # 'wall': local_binary_pattern(wall, N_POINTS, RADIUS, METHOD)
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
    hist(ax4, refs['brick'])
    ax4.set_ylabel('Percentage')

    ax2.imshow(unmarkup)
    ax2.axis('off')
    hist(ax5, refs['grass'])
    ax5.set_xlabel('Uniform LBP values')

    # ax3.imshow(wall)
    # ax3.axis('off')
    # hist(ax6, refs['wall'])

    plt.show()