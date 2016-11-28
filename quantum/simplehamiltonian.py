from matplotlib import pylab, pyplot as plt
import numpy as np

# 3x3
mat33 = np.matrix([[-1.0,  0.0, -0.4],
                   [ 0.0, -0.4, -0.1],
                   [-0.4, -0.1, -0.3]])

# 4x4
mat44 = np.matrix([[-1.0,  0.0,  -0.4,  0.0 ],
                   [ 0.0, -0.4,  -0.1, -0.05],
                   [-0.4, -0.1,  -0.3, -0.1 ],
                   [ 0.0, -0.05, -0.1, -0.2 ]])

# 5x5
mat55 = np.matrix([[-1.0,  0.0,  -0.4,  0.0 ,  0.0 ],
                   [ 0.0, -0.4,  -0.1, -0.05,  0.0 ],
                   [-0.4, -0.1,  -0.3, -0.1 , -0.2 ],
                   [ 0.0, -0.05, -0.1, -0.2 , -0.1 ],
                   [ 0.0,  0.0,  -0.2, -0.1 , -0.15]])


class SimpleHamiltonian:
    """
    A simple Hamiltonian class wherein the Hamiltonian is represented as a
    matrix with a single value for each block
    """
    def __init__(self, mat, name='full'):
        self.hamiltonian = mat
        self.name = name

    def energy(self):
        """
        Return the energy for the given method
        """
        evals, evecs = np.linalg.eigh(self.hamiltonian)

        return evals.min()

    def approx(self, method='full'):
        """
        Return the hamiltonian corresponding to the given method
        """
        method = method.lower()
        hamiltonian = self.hamiltonian.copy()

        if method == 'full' or method == 'full-ci':
            pass
        elif method == 'hf':
            # Wipe out all excited contributions
            hamiltonian[1:, :] = hamiltonian[:,1:] = 0
        elif method == 'cid':
            # Wipe out all singles contributions
            hamiltonian[1, :] = hamiltonian[:, 1] = 0
            # Wipe out everything higher than doubles
            hamiltonian[3:, :] = hamiltonian[:, 3:] = 0
        elif method == 'cisd':
            # Wipe out everything higher than doubles
            hamiltonian[3:, :] = hamiltonian[:, 3:] = 0
        elif method == 'cidt':
            # Wipe out all singles contributions
            hamiltonian[1, :] = hamiltonian[:, 1] = 0
            # Wipe out everything higher than triples
            hamiltonian[4:, :] = hamiltonian[:, 4:] = 0
        elif method == 'cisdt':
            # Wipe out everything higher than triples
            hamiltonian[4:, :] = hamiltonian[:, 4:] = 0
        elif method == 'cidq':
            # Wipe out all singles contributions
            hamiltonian[1, :] = hamiltonian[:, 1] = 0
            # Wipe out all triples contributions
            hamiltonian[3, :] = hamiltonian[:, 3] = 0
            # Wipe out everything higher than quadruples
            hamiltonian[5:, :] = hamiltonian[:, 5:] = 0
        elif method == 'cisdtq':
            # Wipe out everything higher than quadruples
            hamiltonian[5:, :] = hamiltonian[:, 5:] = 0
        else:
            raise Exception('Invalid method: {}'.format(method))

        return SimpleHamiltonian(hamiltonian, name=method)

    def plot(self, ax, cmap='default'):
        if cmap == 'default':
            cmap = plt.get_cmap('Oranges_r')
            #cmap = plt.get_cmap('BuPu_r')

        ax.set_title('{:s}: {: >10.9f}'.format(self.name.upper(), self.energy()))
        im = ax.imshow(self.hamiltonian, interpolation='nearest', cmap=cmap)

        return im


def scale(mat):
    """
    Gradually includes excited contributions and shows their effects on
    eigenvectors and eigenvalues
    """
    # TODO: Use subplots to display together
    # TODO: Plot change in energy (or energies)
    pylab.matshow(mat)
    for i in np.arange(0, 1.1, 0.1):
        hamiltonian = i*mat
        hamiltonian[0, 0] = -1

        print(hamiltonian)

        evals, evecs = np.linalg.eigh(hamiltonian)
        print('Eigen values\n{}\nEigen vectors\n{}'.format(evals,evecs))

        pylab.matshow(hamiltonian)
    pylab.show()


def run(mat):
    """
    Plots heatmaps of various SimpleHamiltonians
    """
    fig, axes = plt.subplots(2, 3)
    # flatten
    axes = axes.reshape(-1)

    h = SimpleHamiltonian(mat)

    methods = ['hf', 'cid', 'cisd', 'cisdt', 'cidq', 'full']
    ims = [h.approx(method).plot(ax) for method, ax in zip(methods, axes)]

    fig.colorbar(ims[-1])

    plt.show()


#run(mat55)
