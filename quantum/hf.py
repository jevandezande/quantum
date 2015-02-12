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
            if type(orb.orbital) != self.orb_type:
                raise SyntaxError("Cannot mix orbital types.")

        self.orbitals = orbitals

    def orbs(self):
        return self.orbitals


class I:
    """
    One-electron energy terms
    """

    def __init__(self, wavefunction):
        self.wfn = wavefunction

        for orb in self.wfn.orbs:
            self.i.append()

    def __str__(self):
        return " + ".join(["I({0},{0})".format(orb) for orb in self.wfn.orbs])

    def spin_integrate(self):
        pass


class J:
    """
    Two-electron Coulomb energy terms
    """

    def __init__(self, wavefunction):
        self.wfn = wavefunction

        for orb in self.wfn.orbs:
            self.i.append()

    def spin_integrate(self):
        pass


class K:
    """
    Two-electron exchange energy terms
    """

    def __init__(self, wavefunction):
        self.wfn = wavefunction

        for orb in self.wfn.orbs:
            self.i.append()

    def spin_integrate(self):
        pass


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
        return "{}\n{}\n{}".format(i, j, k)

    def spin_integrated(self):
        pass
