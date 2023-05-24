# PEN: Process Estimator neural Network for root cause analysis using graph convolution

This repository contains jupyter notebooks to reproduce the results of the following paper:

> Leonhardt, Viktor, Felix Claus, and Christoph Garth. "PEN: Process Estimator neural Network for root cause analysis using graph convolution." Journal of Manufacturing Systems 62 (2022): 886-902, https://doi.org/10.1016/j.jmsy.2021.11.008.

## Dependencies

In order to get this running, the following python libraries are necessary:
* tensorflow
* numpy
* scipy

Optional dependencies are dependent to the data set PEN was design to:
* pandas
* vtk

The data set we used was in VTK format, so an optional dependency to extract or illustrate the data is `vtk`.

## Data
Unfortuneately, we are not allowed to share the data publicly. For the convolution, we need the adjacency matrix of the geometry and the values of the displacement of each part.

## Order of execution
The execution of this toolbox is not linear and some steps need to be repeated in order to adjust the PEN to a data set. At first, one needs to be aware of the properties of the data set, i.e. the spectrum of the training data set. This gives information of the maximal frequency and other hyper-parameters regarding the spectrum. After the training of the PEN, it can be analyzed to improve the PEN further.
<dl>
    <dt>Extract adjacency list.ipynb</dt>
    <dd>Extracts the adjacency matrix from an unstructured grid using `vtk`.</dd>
    <dt>Fourier Transform.ipynb</dt>
    <dd>Calculate the Fourier basis and stores it.</dd>
    <dt>Extract Vertices.ipynb</dt>
    <dd>Extracts the point data from a `vtk` unstructured grid into an `numpy` array.</dd>
    <dt>Visualize Spectrum.ipynb</dt>
    <dd>Visualizes the spectrum of the whole data set to identify the maximal frequency and other parameters related to the spectrum. In addition, the Fourier basis can be illustrated.</dd>
    <dt>PEN.ipynb</dt>
    <dd>The training of the PEN is done in this script.</dd>
    <dt>Analyse PEN.ipynb</dt>
    <dd>After the training, the PEN can be validated and analyzed further using a validation data set.</dd>
    <dt>Visualize Filter.ipynb</dt>
    <dd>Finally, we can illustrate the trained filters of the PEN.</dd>
</dl>
