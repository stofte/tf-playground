"""
Plotting helper module
Modified from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
"""
import math as math
import matplotlib.pyplot as plt # pylint: disable=E0401

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from IPython.core.display import HTML
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt

class Recorder:
    def __modifymarkup(self, markup):
        lines = markup.splitlines()

        return markup

    def render(self, frames):
        """
        Displays a list of frames as a gif, with controls
        """
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        wrapped_anim = display_animation(anim, default_mode='loop')
        # dumps a html fragment
        html = self.__modifymarkup(wrapped_anim._repr_html_())
        print(html)
        display(HTML(html))

class MultiPlot:
    """
    Matplotlib helper module renders multiple plots to be drawn in a single figure using a grid
    Modified from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """
    default_plot_width_inches = 20
    def __init__(self):
        self.images = []
        self.n_images = 0
        self.n_rows = 0
        self.n_cols = 0
        self.titles = []

    def data(self, images, titles):
        """
        Sets the matrixes (and possibly titles) which will be plotted
        """
        self.images = images
        self.n_images = len(images)
        self.n_rows = 0
        self.n_cols = 0
        if titles is None:
            titles = ['Image (%d)' % i for i in range(1, self.n_images + 1)]
        self.titles = titles        

    def render(self, plot_width, plot_height, columns, colorbar):
        """
        Renders the multiple plots within a single figure

        Parameters
        ---------
        plotwidth: width of an individual plot

        columns: number of columns
        """
        self.n_rows = math.ceil(self.n_images / columns)
        self.n_cols = math.ceil(self.n_images / self.n_rows)
        fig = plt.figure()
        fig.set_figwidth(plot_width)
        fig.set_figheight(plot_height + self.default_plot_width_inches * 0.1)

        vmin = min([m.min() for m in self.images])
        vmax = max([m.max() for m in self.images])

        for index, (image, title) in enumerate(zip(self.images, self.titles)):
            axis = fig.add_subplot(self.n_cols, self.n_cols, index + 1, aspect='equal')
            axis.set_title(title)
            image_ref = axis.pcolor(image, cmap='bwr', vmin=vmin, vmax=vmax)
        if colorbar is not None:
            cbar_ax = fig.add_axes(colorbar)
            fig.colorbar(image_ref, fraction=0.046, cax=cbar_ax, cmap='bwr')
        plt.show()
