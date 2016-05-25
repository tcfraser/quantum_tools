import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from shutil import copyfile

# Computer modern fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def ket(t):
    return r"\left|{t}\right\rangle".format(t=t)

def bra(t):
    return r"\left\langle{t}\right|".format(t=t)

def ketbra(t):
    return r"{0}{1}".format(ket(t), bra(t))

labels = [ketbra(r"\Phi^+"), ketbra(r"\Phi^-"), ketbra(r"\Psi^+"), ketbra(r"\Psi^-")]

def ls(scenario):
    return [labels[state] for state in scenario]

def plot_max_entangled_triplets(prefix, scenario):
    filename = '{0}_{1}{2}{3}'.format(prefix, *scenario)

    file = open('solutions/{filename}.csv'.format(filename=filename), 'rb')
    # file = open('solutions/restricted.csv', 'rb')
    data = np.loadtxt(file, delimiter=' ', skiprows=1)

    bar_width = 1

    plt.figure()
    plt.xlabel(r"Inequality Index")
    plt.ylabel(r"Minimal Net Correlation")
    plt.title(r"Inequality Saturation for Bell States $\rho_X = {0}, \rho_Y = {1}, \rho_Z = {2}$".format(*ls(scenario)))
    plt.bar(data[:,0],data[:,1],width=bar_width)
    plt.gca().set_xticks(np.arange(data.shape[0]) + bar_width / 2)
    plt.gca().set_xticklabels(np.arange(1, data.shape[0] + 1))
    plt.savefig('figures/{filename}.pdf'.format(filename=filename), format='pdf', figsize=(4,7))
    plt.show()

def plot(filename, title):
    csv_filename = 'solutions/{filename}.csv'.format(filename=filename)
    copyfile('solutions/temp.csv', csv_filename)
    file = open(csv_filename, 'rb')
    # file = open('solutions/restricted.csv', 'rb')
    data = np.loadtxt(file, delimiter=' ', skiprows=1)

    bar_width = 1

    plt.figure()
    plt.xlabel(r"Inequality Index")
    plt.ylabel(r"Minimal Net Correlation")
    plt.title(title)
    plt.bar(data[:,0],data[:,1],width=bar_width)
    plt.gca().set_xticks(np.arange(data.shape[0]) + bar_width / 2)
    plt.gca().set_xticklabels(np.arange(1, data.shape[0] + 1))
    plt.savefig('figures/{filename}.pdf'.format(filename=filename), format='pdf', figsize=(4,7))
    plt.show()

if __name__ == '__main__':
    # plot(filename='sat_general_neig')
    # plot(filename='impl_strat_1_38', title=r'General States - General Measurements')
    # plot(filename='impl_strat_2_38_5', title=r'Pure States - Pure Measurements - 5 Tries')
    # plot(filename='wut', title=r'Wut')
    plot(filename='basin_hopping_strat_2', title=r'Basin Hopping Max Entangled 10 Tries')
    # plot(filename='impl_strat_1_38_5', title=r'General States - General Measurements - 5 Tries')
    # plot(filename='pure_pure', title=r'Pure States - Pure Measurements {0} =\cos(q_i){1} + \sin(q_i){2}'.format(ket(r'\psi'), ket(r'00'), ket(r'11')))
    # plot_max_entangled_triplets(prefix='max_entangled_neig', scenario=(0,0,0))
    # plot_max_entangled_triplets(prefix='test', scenario=(0,0,0))
    # plot_max_entangled_triplets(prefix='max_entangled_neig', scenario=(0,1,2))