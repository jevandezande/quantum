from matplotlib import pylab, pyplot as plt
import numpy as np


# 6x6
mat66 = -np.matrix([[100,30, 7, 7, 3, 3, 1, 0],
                    [ 30,80, 7, 7, 3, 3, 1, 1],
                    [  7, 7,90,30, 7, 7, 3, 3],
                    [  7, 7,30,70, 7, 7, 3, 3],
                    [  3, 3, 7, 7,75,30, 7, 7],
                    [  3, 3, 7, 7,30,50, 7, 7],
                    [  1, 1, 3, 3, 7, 7,60,30],
                    [  0, 1, 3, 3, 7, 7,30,40]])

class ZHamiltonian:
    """
    A relativistic Hamiltonian class wherein the Hamiltonian is represented as a
    matrix with a single value for each block. The spin-sectors are almost block
    diagonal (designated by sectors with small off-diagonal coupling that can
    be removed using approx()
    """
    def __init__(self, hamiltonian, name='full', sectors=None):
        """
        :param hamiltonian: a matrix of the hamiltonian elements
        :param name: name for the hamiltonian (often the level of approximation)
        :param sectors: the spin-sectors of the hamiltonian
        """

        self.hamiltonian = hamiltonian
        self.name = name
        self.sectors = sectors
        if sectors is None:
            self.sectors = [0]


    def energy(self):
        """
        Return the energy for the given method
        """
        evals, evecs = np.linalg.eigh(self.hamiltonian)

        return evals.min()

    def approx(self, method='full'):
        """
        Return the hamiltonian with the specified elements zeroed out

        e.g. with 1x1 sectors:

        level-1: allows only coupling with adjacent sectors
        -------------     -------------
        | a | b | c |     | a | b | 0 |
        -------------     -------------
        | d | e | f | ==> | d | e | f |
        -------------     -------------
        | g | h | i |     | 0 | f | i |
        -------------     -------------

        1: allows only coupling of the 0th and 1st sectors
        -------------     -------------
        | a | b | c |     | a | b | 0 |
        -------------     -------------
        | d | e | f | ==> | d | e | 0 |
        -------------     -------------
        | g | h | i |     | 0 | 0 | i |
        -------------     -------------
        """
        sectors = self.sectors
        hamiltonian = self.hamiltonian.copy()

        if isinstance(method, str):
            if method.lower() == 'full':
                pass
            elif 'level-' in method:
                m = int(method[6:])
                for i in range(1, len(sectors) - m):
                    hamiltonian[sectors[m + i]:, sectors[i-1]:sectors[i]] = 0
                    hamiltonian[sectors[i-1]:sectors[i], sectors[m + i]:] = 0
                    
                    
        elif isinstance(method, int):
            for a in self.sectors[method + 1:]:
                hamiltonian[a:, 0:a] = hamiltonian[0:a, a:] = 0
        else:
            raise Exception('Invalid method: {}'.format(method))

        return ZHamiltonian(hamiltonian, name=method)

    def plot(self, ax, cmap='default'):
        """
        Makes a heatmap
        """
        if cmap == 'default':
            cmap = plt.get_cmap('inferno')

        ax.set_title('{:}: {: >8.3f}'.format(self.name, self.energy()))
        im = ax.imshow(self.hamiltonian, interpolation='nearest', cmap=cmap)

        return im


def runZ(mat):
    """
    Plots heatmaps of various ZHamiltonians
    """
    methods = ['Full', 0, 1, 2]
    methods = ['Full', 'level-0', 'level-1', 'level-2']
    sectors = [0, 2, 4, 6]

    fig, axes = plt.subplots(1, len(methods))
    # flatten
    axes = axes.reshape(-1)

    h = ZHamiltonian(mat, 'full', sectors)

    ims = [h.approx(method).plot(ax) for method, ax in zip(methods, axes)]

    #fig.colorbar(ims[-1])
    cbaxes = fig.add_axes([0.93, 0.27, 0.02, 0.46]) 
    cb = plt.colorbar(ims[1], cax=cbaxes)

    plt.show()


runZ(mat66)
