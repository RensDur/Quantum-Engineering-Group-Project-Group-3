<div align="center">

**TN3175 Quantum Engineering Group Project (2022/23 Q2)**
# Group 3

<br>

### **AUTHORS**<br>
Arnav Chopra<br>
Jules Drent<br>
Rens Dur<br>
Ion Plamadeala<br>
Oliver Sihlovec

</div>

<br><br>
# About this repository
This repository contains all the files that we used in order to perform all experiments. For each experiment we ran, a separate folder is placed in this repository, where you will find the following files:

1. A Jupyter Notebook that contains the code for the experiment
2. A file or folder that contains all the results that we obtained.

# Quick access
For each experiment, important files are listed in the table below.

| Experiment                    | Important files |
| ----------                    | --------------- |
| Single qubit - two clusters   | [Jupyter Notebook](./SingleQubitExperiment/ClusteringExperimentSingleQubit.ipynb), [Results](./SingleQubitExperiment/results/) |
| $N$ qubits - $2^N$ clusters   | [Jupyter Notebook](./Multiple%20Cluster%20Experiment/Multiple%20Clusters.ipynb), [Results](./Multiple%20Cluster%20Experiment/results/) |
| Multiple Features             | [Jupyter Notebook](./Multifeature%20Experiment/MultifeatureExperiment.ipynb), [Results](./Multifeature%20Experiment/results/) |
| Parallel Qubit Clustering     | [Jupyter Notebook](./Parallel%20Qubit%20Clustering/Parallel%20Qubit%20Clustering%20Batch.ipynb), [Results](./Parallel%20Qubit%20Clustering/results/) |

<br>

The report can be found here: [Report PDF](./Final%20Report%20Group%203.pdf)

<br>

# Abstract
Here we present a quantum data clustering algorithm, using a Variational Quantum Eigensolver (VQE). The algorithm allows us to distinguish data points, with multiple features, into several clusters. The basis of the algorithm is to treat the clustering problem as an optimisation problem and use a VQE to solve it. Our approach relies on encoding each data point onto a Bloch sphere and minimising a cost function in order to optimise the clustering of the data points, into a specified number of classes. We use qubit-states as reference states for each cluster, the relation between these reference states and data points are the parameters that we attempt to optimise in the quantum algorithm. We apply the algorithm to a variety of datasets, such as datasets with multiple features and multiple classes, in order to benchmark the algorithm with an assortment of inputs. We utilise singular qubit systems as well as systems with multiple qubits, for more complex inputs. Furthermore, we parallelise our circuit, in order to compute several data points simultaneously, such that it can be efficiently implemented on quantum hardware. We benchmark our algorithm using real datasets, and find that it is comparable to classical clustering algorithms in terms of accuracy. The algorithm can also be extended to accommodate higher dimensionality.

*Continue reading the report here: [Report PDF](./Final%20Report%20Group%203.pdf)*

<br><br>

<div align="center">

*Delft University of Technology* <br>
*Faculty of Applied Sciences* <br><br>
January 2023

</div>
