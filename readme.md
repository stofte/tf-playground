Tensorflow playground
=====================
    
Uses python 2.7 (PySide doesnt support 3.5, only 3.2, so for this component 2.* is required assuming QT backend is used)

- `mnist_softmax.py`: https://www.tensorflow.org/get_started/mnist/beginners
- `plotting.py`: matplotlib helper

Windows Subsystem for Linux
===========================

Installing matplotlib deps

    sudo apt install cmake gcc g++ qt4-qmake libqt4-dev
    sudo pip install pyside

Ensure the matplotlib config uses the QT4 backend: `Qt4Agg`. Use `python -m site` to see where packages are sourced from to find the install folder. It installed here for my case: 

    /usr/local/lib/python2.7/dist-packages/matplotlib/mpl-data/matplotlibrc
    
Windows folders are mounted under `/mnt/c` and so on for each drive letter.

IPython/Jupyter Notebook

    sudo apt-get install ipython ipython-notebook
    sudo -H pip install jupyter

Then start the notebook with `jupyter notebook` which should start a server serving the notebook itself.

[Xming](https://sourceforge.net/projects/xming/) is a X server for Windows, it'll allow the use of GUI elements from LXSS. Xming can be started anytime after Bash, but setting the following in `~/.bashrc` is also required:

    export DISPLAY=:0


