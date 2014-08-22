# RRS Python Scientific Tools introduction

These notes are at: http://bit.ly/rrs_scipy_notes

- Course notes: http://scipy-lectures.github.io
- [IPython notebook](http://nbviewer.ipython.org/gist/stefanv/3e3a3049a7b245d69f39)

Q&A
---
1. How do you draw lines with varying colour.

See [this example notebook](http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb).

2. Where can I find the Cython lecture notes?

They are hosted [here on GitHub](https://github.com/stefanv/teaching/blob/master/2013_assp_zurich_cython/slides/zurich2012_cython.pdf?raw=true)

3. Where can I find the real-time Bokeh demo?

See the [Bokeh source repository](https://github.com/ContinuumIO/bokeh)

It's under ``bokehjs/demo/spectrogram``.

4. And the Matplotlib oscilloscope?

I've uploaded it to the same
[place as these notes](http://bit.ly/rrs_scipy_notes).

5. Which other projects were mentioned?

- [Pandas](http://pandas.pydata.org)
- [Bokeh](http://bokeh.pydata.org)
- [Numba](http://numba.pydata.org)
- [PyOpenCL](http://mathema.tician.de/software/pyopencl/) (see
  [array example](http://documen.tician.de/pyopencl/))

Also see the [Numpy and SciPy documentation](http://docs.scipy.org/doc/).

6. Do these tools run on the Raspberry Pi?

They do!  Have a look at this [face/mask detection
demo](http://www.aicbt.com/disguise-detection/).

## Original notes

### Editors

 - Spyder (included with Anaconda)

 - [PyCharm](http://www.jetbrains.com/pycharm/)

 - See [configuration instructions for Anaconda](http://docs.continuum.io/anaconda/ide_integration.html)

## Agenda

```
 8:15 - 8:30 : Coffee/tea
 8:30 - 09:00 : Installation check, IPython
 9:00 - 10:30 : The numpy array object
 10:30 - 10:45 : Coffee/tea
 10:45 - 12:30 : Numerical operations on arrays, SciPy
 12:30 - 13:00 : Lunch
 13:00 - 15:00 : Matplotlib, realtime plotting, Bokeh
```

## Introduction

- Workshop content is fully adaptable
- Please steer direction by asking questions along the way

http://scipy-lectures.github.io/intro/intro.html

## IPython

- Notebook (launching, navigation, capabilities)
- [Example CTPUG notebook](http://nbviewer.ipython.org/github/stefanv/teaching/blob/master/2014_ctpug_ipython/ctpug_ipython.ipynb)
- [Slide version of the above](https://rawgit.com/stefanv/teaching/master/2014_ctpug_ipython/ctpug_ipython.slides.html#)

Notable utilities:

- TAB-completion
- ?, ??
- timeit
- !ls
- %run (with inspection afterwards)
- cpaste

## Python

Any questions about the language itself?

## Numpy

http://scipy-lectures.github.io/intro/numpy/array_object.html
http://scipy-lectures.github.io/intro/numpy/operations.html

## Cython

Wrapping C code

## Real-time plotting

- matplotlib -- FuncAnimation
- Bokeh -- spectrogram demo
