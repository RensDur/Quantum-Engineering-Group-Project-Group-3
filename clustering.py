import os
from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt

from qiskit import execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import EfficientSU2

from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI
from qiskit.algorithms.optimizers import ADAM as ADAM
from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit import transpile
from qiskit.quantum_info import state_fidelity
from numpy.linalg import norm
from qiskit.quantum_info.states.quantum_state import QuantumState
from tqdm import tqdm
from qiskit.visualization import plot_bloch_vector

QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')

project_name = 'Qiskit-vqe'
# authentication = get_authentication()
# QI.set_authentication(authentication, QI_URL, project_name=project_name)
qi_backend = Aer.get_backend("aer_simulator")

np.random.seed(999999)
p0 = np.random.random()
target_distr = {0: p0, 1: 1 - p0}

qubits = 1
ref_states = [np.array([1, 0]), np.array([0, 1])]


class Data:

    def __init__(self, cluster_number=2, cluster_size=10):
        self.clusters = []
        self.cluster_number = cluster_number
        self.cluster_size = cluster_size
        self.generate_clusters()

    def generate_clusters(self) -> None:
        mu, sigma = .25, 0.1
        np.random.seed(999999)
        # cluster 1
        x1 = np.random.normal(mu, sigma, self.cluster_size)
        # y1 = np.random.normal(mu, sigma, 10)

        # cluster 2
        mu, sigma = .75, 0.1
        x2 = np.random.normal(mu, sigma, self.cluster_size)
        # y2 = np.random.normal(mu, sigma, 10)

        # multiply the data with pi to get the angle on the bloch sphere
        x1 = x1 * np.pi
        x2 = x2 * np.pi
        self.clusters = [x1, x2]
        self.plot_clusters(self.clusters)

    def get_full_dataset(self) -> np.array:
        return np.concatenate(self.clusters)

    @staticmethod
    def get_cluster_for_point(point, reference_states) -> int:
        """
        :param point: the point represented by a state-vector for which we want to find the cluster
        :param reference_states: the reference states for each cluster
        :return: the cluster for the point
        """
        fidelities = np.array([state_fidelity(point, ref_i) for ref_i in reference_states])
        print(fidelities)
        return np.argmax(fidelities)

    def plot_clusters(self, clusters: List[np.array]) -> None:
        plt.figure(figsize=(10, 10))
        for i in range(self.cluster_number):
            plt.hist(clusters[i], bins=20, alpha=0.5, label='cluster {}'.format(i))
        plt.legend()
        plt.show()

    def get_clusters_for_dataset(self, point_transformation: Callable[[np.array], np.array],
                                 reference_vectors: List[np.array]) -> List[np.array]:
        """
        :param point_transformation: a lambda function that computes the output state vector from the input datapoint
        :param reference_vectors: the reference vectors for each cluster
        :return: the clusters for the dataset
        """
        clusters = [list() for _ in range(self.cluster_number)]
        for point in self.get_full_dataset():
            print("for point {}".format(point), end=' fidelity is: ')
            clusters[self.get_cluster_for_point(point_transformation(point), reference_vectors)].append(point)

        # plot the clusters
        self.plot_clusters(clusters)
        return clusters


dataset = Data(cluster_size=10)
dps = dataset.get_full_dataset()


def get_var_form(initial_params, optim_params, qubits):
    qr = QuantumRegister(qubits, name="q")
    # cr = ClassicalRegister(qubits, name="c")
    qc = QuantumCircuit(qr)
    bind_dict = {}
    for i in range(qubits):
        qc.u(initial_params[i], initial_params[i], 0, qr[i])
    for i in range(qubits - 1):
        qc.u(optim_params[i][0], optim_params[i][1], optim_params[i][2], qr[i])
        qc.cx(i, i + 1)
    qc.u(0, optim_params[qubits - 1][1], 0, qr[qubits - 1])
    qc = qc.bind_parameters(bind_dict)
    # qc.measure_all()
    qc.save_statevector()

    # qc.draw("mpl")
    return qc


def small_h(state_1, state_2, reference_states) -> np.array:
    """
    f_{ij} = f(state_1, reference_states[i]) * f(state_2, reference_states[j])
    :param state_1: the output state for the first point
    :param state_2: the output state for the second point
    :param reference_states: the reference states for each cluster
    :return: the fidelity product array
    """

    fidelity_1 = np.array([state_fidelity(state_1, ref_i) for ref_i in reference_states])
    fidelity_2 = np.array([state_fidelity(state_2, ref_i) for ref_i in reference_states])
    return np.sum(np.multiply(fidelity_1, fidelity_2))


count = 1


def objective_function(params):
    """Compares the output distribution of our circuit with
    parameters `params` to the target distribution."""
    # Create circuit instance with paramters and simulate it
    global count
    cost = 0
    for i in tqdm(range(len(dps))):
        for j in range(len(dps)):
            qc_i = transpile(get_var_form([dps[i]], params, qubits), backend=qi_backend)
            result_i = qi_backend.run(qc_i).result()

            qc_j = transpile(get_var_form([dps[j]], params, qubits), backend=qi_backend)
            result_j = qi_backend.run(qc_j).result()
            cost += norm(np.array([dps[i]]), np.array([dps[j]])) * small_h(result_i.get_statevector(),
                                                                           result_j.get_statevector(), ref_states)
    print(count)
    print(cost)
    count += 1
    return cost


optimizer = ADAM(maxiter=5, tol=1e-06)

# Create the initial parameters (noting that our
# single qubit variational form has 3 parameters)
params = [np.random.rand(3) for i in range(qubits)]
result = optimizer.minimize(
    fun=objective_function,
    x0=params)


# Obtain the output distribution using the final parameters
def run_for_one_datapoint(params, point):
    qc = transpile(get_var_form([point], params, qubits), backend=qi_backend)
    result = qi_backend.run(qc).result()
    return result.get_statevector()


if __name__ == '__main__':
    dataset.get_clusters_for_dataset(lambda x: run_for_one_datapoint(result.x, x), ref_states)
