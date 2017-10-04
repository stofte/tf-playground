import math as math
import numpy as np
import matplotlib.pyplot as plt

# https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, titles=None, figwidth=3, cols=1, colorbar=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.

    figwidth: Width of plot in inches

    colorbar: 4 element list for positioning the colorbar
    """
    assert(titles is None) or (len(images) == len(titles))
    n_images = len(images)
    n_cols = math.ceil(n_images / cols)
    n_rows = math.ceil(n_images / n_cols)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure()
    fig.set_figwidth(figwidth * n_cols)
    fig.set_figheight(figwidth * (n_rows + 0.2))

    vmax = max([m.max() for m in images])
    vmin = min([m.min() for m in images])

    for index, (image, title) in enumerate(zip(images, titles)):
        axis = fig.add_subplot(cols, np.ceil(n_images/float(cols)), index + 1, aspect='equal')
        axis.set_title(title)
        image_ref = axis.pcolor(image, cmap='bwr', vmin=vmin, vmax=vmax)

    if colorbar is not None:
        cbar_ax = fig.add_axes(colorbar)
        fig.colorbar(image_ref, cax=cbar_ax, cmap='bwr')

    plt.show()
