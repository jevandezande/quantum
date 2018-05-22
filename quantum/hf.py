from abc import abstractmethod


class Wavefunction:
    def __init__(self, *orbitals):
        """
        A combination of Orbitals

        :param orbitals: Orbital objects

        Warning, does not check for duplicate orbitals
        """

        if len(orbitals) == 0:
            raise SyntaxError('Cannot make an empty wavefunction.')

        self.orb_type = type(orbitals[0])
        for orb in orbitals:
            if type(orb) != self.orb_type:
                raise SyntaxError('Cannot mix orbital types.')

        self.orbitals = orbitals

    def __iter__(self):
        for orbital in self.orbitals:
            yield orbital

    def __str__(self):
        return ' '.join(map(str, self))


class ETerms:
    def __init__(self, wavefunction):
        """
        Energy terms
        :param wavefunction: a wavefunction object
        """
        self.wfn = wavefunction


class I(ETerms):
    """
    One-electron energy terms
    """
    def __str__(self):
        return ' + '.join([f'I({orb},{orb})' for orb in self.wfn])

    def spin_integrate(self):
        spin_int_list = []
        for orb in self.wfn:
            spin_int_list.append('I({0},{0})'.format(orb.spatial_str()))
        return ' + '.join(spin_int_list)


class TwoElectron(ETerms):
    def __init__(self, wavefunction):
        """
        Two-electron energy terms
        :param wavefunction: a wavefunction object
        """
        super().__init__(wavefunction)

        self.terms = []
        for i, orb_i in enumerate(self.wfn, start=1):
            for j, orb_j in enumerate(self.wfn.orbitals[i:], start=i):
                self.terms.append((orb_i, orb_j))

        self.spin_int = ''

    @abstractmethod
    def __str__(self):
        pass

    def __iter__(self):
        for term in self.terms:
            yield term

    def spin_integrate(self, k=False):
        """
        Perform spin integration
        :param k: flip sign and remove terms with unmatched spins
        """
        sign = '' if not k else '-'
        spin_int = []
        for i, j in self:
            if not k or i.spin == j.spin:
                spin_int.append(f'{sign}{type(self).__name__}({i.spatial_str()},{j.spatial_str()})')

        self.spin_int = spin_int

        return f' + '.join(spin_int)


class J(TwoElectron):
    """
    Coulomb energy terms
    """
    def __str__(self):
        return ' + '.join([f'J({i},{j})' for i, j in self])


class K(TwoElectron):
    """
    Exchange energy terms
    """
    def __str__(self):
        return ' '.join([f'-K({i},{j})' for i, j in self])

    def spin_integrate(self):
        return super().spin_integrate(True)


class HF:
    def __init__(self, wavefunction):
        """
        Hartree-Fock object, takes a wavefunction.
        Generates the HF energy expression and performs the spin integration
        """
        self.wfn = wavefunction

    def __str__(self):
        i = str(I(self.wfn))
        j = str(J(self.wfn))
        k = str(K(self.wfn))
        return '\n +'.join(terms for terms in (i, j, k) if terms)

    def spin_integrate(self):
        i = I(self.wfn).spin_integrate()
        j = J(self.wfn).spin_integrate()
        k = K(self.wfn).spin_integrate()
        self.spin_int = (i, j, k)
        return '\n +'.join(terms for terms in map(str, (i, j, k)) if terms)
