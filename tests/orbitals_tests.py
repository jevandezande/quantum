from nose.tools import *
from numpy.testing import assert_equal as np_assert_equal

from quantum.orbitals import *


def setup():
    pass


def teardown():
    pass


def test_atomic_spinorbital():
    one_s1a = AtomicSpinOrbital(n=1, l=0, ml=0, spin=1)
    assert_equal(one_s1a.__repr__(), '1sa')
    five_g2b = AtomicSpinOrbital(n=5, l=4, ml=2, spin='beta')
    assert_equal(five_g2b.__repr__(), '5g_{2}b')
    a = one_s1a
    assert_raises(SyntaxError, AtomicSpinOrbital.__init__, a, 2, 2, 1, 1)
    assert_raises(SyntaxError, AtomicSpinOrbital.__init__, a, 2, 1, 2, 1)
    assert_raises(SyntaxError, AtomicSpinOrbital.__init__, a, 2, 1, 1, 'q')


def test_diatomic_spinorbital():
    one_s1a = DiatomicSpinOrbital(n=1, l=0, ml=0, spin=1)
    assert_equal(one_s1a.__repr__(), '1σa')
    five_g2b = DiatomicSpinOrbital(n=1, l=4, ml=-4, spin='beta')
    assert_equal(five_g2b.__repr__(), '1γ_{-4}b')
    a = one_s1a
    assert_raises(SyntaxError, DiatomicSpinOrbital.__init__, a, 2, 1, 2, 1)
    assert_raises(SyntaxError, DiatomicSpinOrbital.__init__, a, 2, 1, 1, 'q')
    assert_raises(SyntaxError, DiatomicSpinOrbital.__init__, a, 2, 4, 2, 'q')


