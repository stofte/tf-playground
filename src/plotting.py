import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math as math
from matplotlib.colors import LinearSegmentedColormap

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

# https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None, figwidth=3):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    
    figwidth: Width of plot in inches
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    n_cols = math.ceil(n_images / cols)
    n_rows = math.ceil(n_images / n_cols)
    figure_width = figwidth * n_cols
    figure_height = figwidth * (n_rows + 0.2)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    fig.set_figwidth(figure_width)
    fig.set_figheight(figure_height)
    
    for n, (image, title) in enumerate(zip(images, titles)):
        axis = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1, aspect='equal')
        axis.set_title(title)
        im = axis.pcolor(image, cmap=blue_red1)

    cbar_ax = fig.add_axes([0.93, 0.145, 0.02, 0.715])
    
    fig.colorbar(im, cax=cbar_ax, cmap=blue_red1)

    plt.show()
