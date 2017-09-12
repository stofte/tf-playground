Tensorflow playground
=====================

Uses python 2.7

Windows Subsystem for Linux
===========================

Installing matplotlib deps

    sudo apt install cmake gcc g++ qt4-qmake libqt4-dev
    sudo pip install pyside

Ensure the matplotlib config uses the QT4 backend: `Qt4Agg`. Use `python -m site` to see where packages are sourced from to find the install folder. It installed here for my case: 

    /usr/local/lib/python2.7/dist-packages/matplotlib/mpl-data/matplotlibrc
    
Windows folders are mounted under `/mnt/c` and so on for each drive letter.
