import numpy as np
from ..config import *
from ..contexts.measurement import Measurement
from ..contexts.quantum_context import QuantumContext
from ..contexts.quantum_context import QuantumProbDist
from ..contexts.state import State
from ..statistics.variable import RandomVariableCollection
from ..utilities import utils
from ..visualization import plot_cloud
from pprint import pprint

perm = utils.get_triangle_permutation()
OUTPUT_FILE_NAME = 'symmetric_quantum_cloud_data.csv'

def uniform_symmetric_qpd():
    rvc = RandomVariableCollection.new(names=['A', 'B', 'C'], outcomes=[2, 2, 2])

    A = Measurement.Strats.Random.seesaw(dim=4, count=2)
    B = C = A

    rhoAB = State.Strats.Random.pure_uniform(4)
    rhoBC = rhoAC = rhoAB

    qc = QuantumContext(random_variables=rvc, measurements=(A,B,C), states=(rhoAB, rhoBC, rhoAC), permutation=perm)
    pd = QuantumProbDist(qc)
    return pd

def generate_data(num):
    data_array = []
    for i in range(num):
        pd = uniform_symmetric_qpd()
        data = list(pd.canonical_ravel())
        data_array.append(data)
    log_to_file(data_array)
    print("Generated {num} behaviours.".format(num=num))

def log_to_file(data_array):
    with open(OUTPUT_DIR + OUTPUT_FILE_NAME, 'w+') as file_:
        for data in data_array:
            data_string = ' '.join(map(str, data))
            file_.write(data_string)
            file_.write("\n")

def get_characters_from_file(num):
    data_array = np.genfromtxt(OUTPUT_DIR + OUTPUT_FILE_NAME, delimiter=" ")

    p000 = 0
    p001 = 1
    p011 = 3
    p111 = 7
    characters = [p000, p001, p111]

    # Get the representative elements
    reduced_data_array = data_array[:, characters]
    if num>0:
        rows = reduced_data_array.shape[0]
        choices = np.random.choice(rows, num, replace=False)
        # print(choices)
        reduced_data_array = reduced_data_array[choices, :]
        # print(reduced_data_array.shape)
    return reduced_data_array

def plot(num):
    col_data = get_characters_from_file(num)
    plot_cloud.plot_3d_cloud(
        col_data,
        title=r'Symmetric Quantum Distributions P(a,b,c)',
        labels=(r'P(0,0,0)', r'P(0,0,1)', r'P(1,1,1)')
    )

if __name__ == '__main__':
    # generate_data(num=500)
    plot(num=10000)
