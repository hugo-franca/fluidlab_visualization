import matplotlib.pyplot as plt
from pyfonts import load_google_font
import matplotlib.font_manager as fm
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.gridspec as gridspec

DEFAULT_FIGSIZE = 2.0*np.array([6.69423, 4.5])
DEFAULT_FONTSIZE = 25

class FluidLabFigure:
    def __init__(self, fontsize=DEFAULT_FONTSIZE, usetex=False, 
                 subplots: ArrayLike | str | None = None, 
                 darkmode=False, *args, **kwargs):
        figsize = np.array(kwargs['figsize']) if 'figsize' in kwargs.keys() else DEFAULT_FIGSIZE
        figsize[0] = DEFAULT_FIGSIZE[0] if figsize[0]==None else figsize[0]
        figsize[1] = DEFAULT_FIGSIZE[1] if figsize[1]==None else figsize[1]
        kwargs['figsize'] = tuple( figsize )
        if( darkmode ):
            kwargs['facecolor'] = 'black'
        self.darkmode = darkmode
        self.usetex = usetex

        self.gridspec = None

        plt.rcParams.update({
            "text.usetex": usetex,
            "font.size": fontsize,
            "text.color": "white" if darkmode else "black",
            "ytick.color": "white" if darkmode else "black",
            "xtick.color": "white" if darkmode else "black",
            "axes.labelcolor": "white" if darkmode else "black",
            "axes.edgecolor": "white" if darkmode else "black",
            'axes.linewidth': 1
        })

        self._matplotlib_fig = plt.figure(*args, **kwargs)

        if( subplots ):
             
            if( isinstance(subplots, str) ):
                self.subplots = {}

            elif( np.issubdtype(np.array(subplots).dtype, np.integer) ):
                subplots = np.array(subplots)

                self.subplots = []
                for i in range(subplots[0]*subplots[1]):
                    self.subplots.append( self.add_subplot(subplots[0], subplots[1], i+1) )
                self.subplots = np.atleast_2d( np.squeeze( np.reshape(self.subplots, subplots) ) )
            else:
                print("Unexpected dtype: ", subplots.dtype)
    
    def add_subplot(self, *args, **kwargs):
        new_ax = self._matplotlib_fig.add_subplot(*args, **kwargs)

        if( self.darkmode ):
            new_ax.set_facecolor("black")
            new_ax.xaxis.label.set_color('white')
            new_ax.tick_params(axis='x', colors='white')
            new_ax.yaxis.label.set_color('white')
            new_ax.tick_params(axis='y', colors='white')
            for axis in ['top', 'bottom', 'left', 'right']:
                new_ax.spines[axis].set_color('white')

        return new_ax
    
    def __getattr__(self, name):
        # Delegate unknown attributes to the wrapped Figure
        return getattr(self._matplotlib_fig, name)


def set_global_font(font_name: str = "Baloo Tamma 2"):
    # Setting up default font
    font = load_google_font(font_name)
    font_path = str(font).split("file=")[1].split(".ttf")[0] + ".ttf"
    font_path = font_path.replace(r"\\", r"/")
    font_path = font_path.replace(r"\:", r":")
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.rcParams['mathtext.it'] = 'Baloo Tamma 2'
    plt.rcParams['mathtext.rm'] = 'Baloo Tamma 2'

def get_global_font():
    return plt.rcParams['font.family']


def sub_gridspec(*args, **kwargs):
    return gridspec.GridSpecFromSubplotSpec(*args, **kwargs)