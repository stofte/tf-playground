Tensorflow playground (Windows)
===============================

Recommended to use Anaconda for managing installation of components. Repo tested with Python 3.6.4. Currently [WSL does not support exposing GPU](https://github.com/Microsoft/WSL/issues/1788), limiting what use can be done usefully in the WSL Bash console on Windows. Windows/x64 specific Tensorflow [wheels](https://www.python.org/dev/peps/pep-0427/) for [Python can be found here](https://github.com/fo40225/tensorflow-windows-wheel).

Notes:

 - Code tested against v. 1.7.0/py36//GPU/cuda91cudnn71avx2 (`tensorflow_gpu-1.7.0-cp36-cp36m-win_amd64.whl`)
 - Measure perf on Windows: `powershell -Command "Measure-Command { python script.py | Out-Default}"`
 - `jupyter notebook` for launching Jupyter
 