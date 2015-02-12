from abc import abstractmethod


class Wavefunction:
    """A combination of Orbitals"""

    def __init__(self, *orbitals):
        """
        Make a wavefunction
        Warning, does not check for duplicate orbitals

        :param orbitals: Orbital objects
        """

        if len(orbitals) == 0:
            raise SyntaxError("Cannot make an empty wavefunction.")

        self.orb_type = type(orbitals[0])
        for orb in orbitals:
            if type(orb) != self.orb_type:
                raise SyntaxError("Cannot mix orbital types.")

        self.orbitals = orbitals

    def __str__(self):
        return " ".join(map(str, self.orbitals))

    def close(self):
        """
        Find all closed shells and return string accordingly
        """
        pass

    def orbs(self):
        return self.orbitals


class I:
    """
    One-electron energy terms
    """

    def __init__(self, wavefunction):
        self.wfn = wavefunction

    def __str__(self):
        return " + ".join(["I({0},{0})".format(orb) for orb in self.wfn.orbs()])

    def spin_integrate(self):
        spin_int_list = []
        for orb in self.wfn.orbs():
            spin_int_list.append("I({0},{0})".format(orb.spatial_str()))
        return " + ".join(spin_int_list)


class TwoElectron:
    """
    Two-electron energy terms
    """
    def __init__(self, wavefunction):
        self.wfn = wavefunction

        self.terms = []
        for i, orb_i in enumerate(self.wfn.orbs()):
            for j, orb_j in enumerate(self.wfn.orbs()[i + 1:], start=i+1):
                self.terms.append((orb_i, orb_j))

    @abstractmethod
    def __str__(self):
        pass

    def spin_integrate(self, match_spin, name, group=False):
        """
        Perform spin integration
        """
        spin_int = []
        for i, j in self.terms:
            if not match_spin or i.spin == j.spin:
                spin_int.append("{}({},{})".format(name,
                                                   i.spatial_str(),
                                                   j.spatial_str()))

        return " + ".join(spin_int)


class J(TwoElectron):
    """
    Coulomb energy terms
    """
    def __str__(self):
        return " + ".join(["J({},{})".format(i, j) for i, j in self.terms])

    def spin_integrate(self):
        return super().spin_integrate(False, 'J')


class K(TwoElectron):
    """
    Exchange energy terms
    """
    def __str__(self):
        return " + ".join(["K({},{})".format(i, j) for i, j in self.terms])

    def spin_integrate(self):
        return super().spin_integrate(True, 'K')


class HF:
    """
    Hartree-Fock object, takes a wavefunction.
    Generates the HF energy expression, performs the spin integration, and
    groups terms.
    """

    def __init__(self, wavefunction):
        self.wfn = wavefunction

    def __str__(self):
        i = I(self.wfn)
        j = J(self.wfn)
        k = K(self.wfn)
        return "{}\n+ {}\n+ {}".format(i, j, k)

    def spin_integrated(self):
        pass
