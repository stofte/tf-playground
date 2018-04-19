Tensorflow playground (Windows)
===============================

Recommended to use Anaconda for managing installation of components. Repo tested with Python 3. Currently [WSL does not support exposing GPU](https://github.com/Microsoft/WSL/issues/1788), limiting what use can be done usefully in the WSL Bash console on Windows. Windows/x64 specific Tensorflow [wheels](https://www.python.org/dev/peps/pep-0427/) for [Python can be found here](https://github.com/fo40225/tensorflow-windows-wheel).

MNIST dataset

 - [Softmax regression](src/mnist_softmax.ipynb)
 - [Convolutional neural net](src/mnist_convnn.ipynb)
 - [Convnet using Keras](src/mnist_convnet_keras.ipynb)

Software/library versions:

 - Windows 10 (v1709)
 - CUDA 9.1.85
 - python 3.6.4
 - anaconda 4.4.10
 - jupyter 1.0
 - tensorflow 1.7
 - gym 0.10.5
 - [GraphViz 2.38](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) (Keras visualization)

Notes:

 - Measure perf on Windows: `powershell -Command "Measure-Command { python script.py | Out-Default}"`

TODOS
-----

 - 5-fold validation relevant?
 - [Progress bars?](https://github.com/bstriner/keras-tqdm)
