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
    wfn3 = Wavefunction(DiatomicSpinOrbital(1, 0,  0,  1),
                        DiatomicSpinOrbital(1, 0,  0, -1),
                        DiatomicSpinOrbital(2, 1, -1, -1))

    assert_equal(str(wfn1), "1sa")
    assert_equal(str(wfn2), "1sa 1sb 2p_{-1}b")
    assert_equal(str(wfn3), "1σa 1σb 2π_{-1}b")


def test_i():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    wfn3 = Wavefunction(DiatomicSpinOrbital(1, 0,  0,  1),
                        DiatomicSpinOrbital(1, 0,  0, -1),
                        DiatomicSpinOrbital(2, 1, -1, -1))

    i1 = I(wfn1)
    i2 = I(wfn2)
    i3 = I(wfn3)

    assert_equal(str(i1), "I(1sa,1sa)")
    assert_equal(str(i2), "I(1sa,1sa) + "
                          "I(1sb,1sb) + "
                          "I(2p_{-1}b,2p_{-1}b)")
    assert_equal(str(i3), "I(1σa,1σa) + "
                          "I(1σb,1σb) + "
                          "I(2π_{-1}b,2π_{-1}b)")
    assert_equal(i1.spin_integrate(), "I(1s,1s)")
    assert_equal(i2.spin_integrate(), "I(1s,1s) + "
                                      "I(1s,1s) + "
                                      "I(2p_{-1},2p_{-1})")
    assert_equal(i3.spin_integrate(), "I(1σ,1σ) + "
                                      "I(1σ,1σ) + "
                                      "I(2π_{-1},2π_{-1})")


def test_j():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    j1 = J(wfn1)
    j2 = J(wfn2)
    assert_equal(str(j1), "")
    assert_equal(str(j2), "J(1sa,1sb) + "
                          "J(1sa,2p_{-1}b) + "
                          "J(1sb,2p_{-1}b)")
    assert_equal(j1.spin_integrate(), "")
    assert_equal(j2.spin_integrate(), "J(1s,1s) + "
                                      "J(1s,2p_{-1}) + "
                                      "J(1s,2p_{-1})")


def test_k():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    k1 = K(wfn1)
    k2 = K(wfn2)
    assert_equal(str(k1), "")
    assert_equal(str(k2), "-K(1sa,1sb) "
                          "-K(1sa,2p_{-1}b) "
                          "-K(1sb,2p_{-1}b)")
    assert_equal(k1.spin_integrate(), "")
    assert_equal(k2.spin_integrate(), "-K(1s,2p_{-1})")


def test_HF():
    wfn1 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1))
    wfn2 = Wavefunction(AtomicSpinOrbital(1, 0,  0,  1),
                        AtomicSpinOrbital(1, 0,  0, -1),
                        AtomicSpinOrbital(2, 1, -1, -1))
    hf1 = HF(wfn1)
    hf2 = HF(wfn2)

    assert_equal(str(hf1), "I(1sa,1sa)")
    assert_equal(str(hf2), "I(1sa,1sa) + "
                           "I(1sb,1sb) + "
                           "I(2p_{-1}b,2p_{-1}b)\n +"
                           "J(1sa,1sb) + "
                           "J(1sa,2p_{-1}b) + "
                           "J(1sb,2p_{-1}b)\n +"
                           "-K(1sa,1sb) "
                           "-K(1sa,2p_{-1}b) "
                           "-K(1sb,2p_{-1}b)")

    assert_equal(hf1.spin_integrate(), "I(1s,1s)")
    assert_equal(
        hf2.spin_integrate(),
        "I(1s,1s) + "
        "I(1s,1s) + "
        "I(2p_{-1},2p_{-1})\n +"
        "J(1s,1s) + "
        "J(1s,2p_{-1}) + "
        "J(1s,2p_{-1})\n +"
        "-K(1s,2p_{-1})")
