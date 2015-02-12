from nose.tools import *
from numpy.testing import assert_equal as np_assert_equal

from quantum.orbitals import AtomicSpinOrbital, DiatomicSpinOrbital
from quantum.hf import *


def setup():
    pass


def teardown():
    pass


def test_wavefunction():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0, 0, 1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    assert_equal(str(wfn1), "1s_{0}a")
    assert_equal(str(wfn2), "1s_{0}a 1s_{0}b 2p_{-1}b")


def test_i():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0, 0, 1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    i1 = I(wfn1)
    i2 = I(wfn2)
    assert_equal(str(i1), "I(1s_{0}a,1s_{0}a)")
    assert_equal(str(i2), "I(1s_{0}a,1s_{0}a) + "
                          "I(1s_{0}b,1s_{0}b) + "
                          "I(2p_{-1}b,2p_{-1}b)")
    assert_equal(i1.spin_integrate(), "I(1s_{0},1s_{0})")
    assert_equal(i2.spin_integrate(), "I(1s_{0},1s_{0}) + "
                                      "I(1s_{0},1s_{0}) + "
                                      "I(2p_{-1},2p_{-1})")


def test_j():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0, 0, 1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    j1 = J(wfn1)
    j2 = J(wfn2)
    assert_equal(str(j1), "")
    assert_equal(str(j2), "J(1s_{0}a,1s_{0}b) + "
                          "J(1s_{0}a,2p_{-1}b) + "
                          "J(1s_{0}b,2p_{-1}b)")
    assert_equal(j1.spin_integrate(), "")
    assert_equal(j2.spin_integrate(), "J(1s_{0},1s_{0}) + "
                                      "J(1s_{0},2p_{-1}) + "
                                      "J(1s_{0},2p_{-1})")


def test_k():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0, 0, 1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    k1 = K(wfn1)
    k2 = K(wfn2)
    assert_equal(str(k1), "")
    assert_equal(str(k2), "K(1s_{0}a,1s_{0}b) + "
                          "K(1s_{0}a,2p_{-1}b) + "
                          "K(1s_{0}b,2p_{-1}b)")
    assert_equal(k1.spin_integrate(), "")
    assert_equal(k2.spin_integrate(), "K(1s_{0},2p_{-1})")
