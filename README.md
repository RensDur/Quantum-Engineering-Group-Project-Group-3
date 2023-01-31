<div align="center">

<p style="font-size: 30pt; font-weight: 700; margin: 0; padding: 0;">Quantum Clustering Algorithm</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">APPROACHING UNSUPERVISED CLUSTERING AS AN OPTIMISATION PROBLEM</p>

<p style="font-size: 17pt; font-weight: 400; margin: 0; padding: 0; margin-top: 50px;">Group 3</p>
<p style="font-size: 12.5pt; font-weight: 400; margin: 0; padding: 0;">TN3175 Quantum Engineering Group Project</p>
<p style="font-size: 10pt; font-weight: 100; margin: 0; padding: 0; font-style: italic;">2022/23 Q2</p>

<p style="font-size: 17pt; font-weight: 400; margin: 0; padding: 0; margin-top: 50px;">Authors</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">Arnav Chopra</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">Jules Drent</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">Rens Dur</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">Ion Plamadeala</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0;">Oliver Sihlovec</p>

<hr style="margin-top:50px; margin-bottom: 50px;">

</div>


# About this repository
This repository contains all the files that we used in order to perform all experiments. For each experiment we ran, a separate folder is placed in this repository, where you will find the following files:

1. A Jupyter Notebook that contains the code for the experiment
2. A file or folder that contains all the results that we obtained.
3. A README that contains references to those sections in the report where that specific experiment is explained. The results will also be covered in the report.

# Quick access
For each experiment, important files are listed in the table below.

| Experiment                    | Important files |
| ----------                    | --------------- |
| Single qubit - two clusters   |  |
| $N$ qubits - $2^N$ clusters   |  |
| Multiple Features             |  |
| Parallel Qubit Clustering     | [Jupyter Notebook](./Parallel%20Qubit%20Clustering/Parallel%20Qubit%20Clustering%20Batch.ipynb) |

<div style="height:20px;"></div>

The report can be found here: [Report PDF]()

<div style="height:20px;"></div>

# Abstract
Here we present a quantum data clustering algorithm, using a Variational Quantum Eigensolver (VQE). The algorithm allows us to distinguish data points, with multiple features, into several clusters. The basis of the algorithm is to treat the clustering problem as an optimisation problem and use a VQE to solve it. Our approach relies on encoding each data point onto a Bloch sphere and minimising a cost function in order to optimise the clustering of the data points, into a specified number of classes. We use qubit-states as reference states for each cluster, the relation between these reference states and data points are the parameters that we attempt to optimise in the quantum algorithm. We apply the algorithm to a variety of datasets, such as datasets with multiple features and multiple classes, in order to benchmark the algorithm with an assortment of inputs. We utilise singular qubit systems as well as systems with multiple qubits, for more complex inputs. Furthermore, we parallelise our circuit, in order to compute several data points simultaneously, such that it can be efficiently implemented on quantum hardware. We benchmark our algorithm using real datasets, and find that it is comparable to classical clustering algorithms in terms of accuracy. The algorithm can also be extended to accommodate higher dimensionality.





<div align="center" style="margin-top: 50px;">

<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0; font-style: italic;">Delft University of Technology</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0; font-style: italic;">Faculty of Applied Sciences</p>
<p style="font-size: 12.5pt; font-weight: 100; margin: 0; padding: 0; margin-top: 10px;">January 2023</p>

</div>