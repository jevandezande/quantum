from sympy import assoc_laguerre as L, exp, factorial, pi as π, sqrt, symbols
from sympy.functions.special.spherical_harmonics import Ynm


"""
Produces Hydrogen atom wavefunctions
"""


def check(n: int, l: int, ml: int, Z: int = 1):
    if n < 1:
        raise SyntaxError(f"Shells must be ≥ 1, got: {n=}")
    if not (0 <= l <= n - 1):
        raise SyntaxError(
            f"Azimuthal quantum number (l) must be 0 <= l <= n - 1, got: {l=}"
        )
    if l < abs(ml):
        raise SyntaxError(
            "The magnetic qunatum number (ml) must be -l <= ml <= l, got: {ml=}, {l=}"
        )
    if Z < 1:
        raise SyntaxError("Z must be greater than 0, got: {Z=}")


def psi(n: int, l: int, ml: int, Z: int = 1):
    check(n, l, ml, Z)

    a0, r, θ, φ = symbols("a_0 r θ φ")
    ρ = 2 * r / (n * a0)

    R = (
        sqrt(2 * Z / n / a0) ** 3
        * factorial(n - l - 1)
        / (2 * n * factorial(n + l))
        * exp(-ρ / 2)
        * ρ ** l
    )
    lag = L(2 * l + 1, n - l - 1, ρ)

    Y = Ynm(ml, l, θ, φ) if l else 1

    return 1 / sqrt(π) * R * lag * Y


if __name__ == "__main__":
    # Failures
    for n, l, ml, mass in [(0, 0, 0, 0), (1, 1, 0, 1), (5, 4, 3, 0)]:
        try:
            psi(n, l, ml, mass)
        except SyntaxError:
            pass

    print(psi(1, 0, 0))
    print(psi(2, 0, 0))
    print(psi(2, 1, 0))
    print(psi(2, 1, 1))
