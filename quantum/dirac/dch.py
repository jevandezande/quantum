import numpy as np
import sympy
from sympy import (
    BlockMatrix,
    Function,
    I,
    Matrix,
    diff,
    exp,
    eye,
    simplify,
    symbols,
    zeros,
)
from sympy.physics.quantum import hbar as ħ

sympy.init_printing(use_unicode=True)


def bm(m):
    """Simple block formatted matrix constructor, returns a Matrix"""
    return Matrix(BlockMatrix(m))


def dot(a, b):
    if len(a) == len(b):
        return sum([w * q for w, q in zip(a, b)], zeros(len(a[0])))
    else:
        raise Exception("a and b are of different dimension.")


σ1 = Matrix([[0, 1], [1, 0]])
σ2 = Matrix([[0, I], [-I, 0]])
σ3 = Matrix([[1, 0], [0, -1]])
σ = [σ1, σ2, σ3]
z2 = zeros(2)
α1 = bm([[z2, σ1], [σ1, z2]])
α2 = bm([[z2, σ2], [σ2, z2]])
α3 = bm([[z2, σ3], [σ3, z2]])
α = [α1, α2, α3]

I2 = eye(2)
β = bm([[I2, z2], [z2, -I2]])

# Particle at Rest (5.3.1)
c, t, me = symbols("c t m_e", real=True)

mat = Matrix.eye(4)
solutions = [
    exp(sign * I * me * c ** 2 / ħ * t) * mat.col(i)
    for i, sign in enumerate([-1] * 2 + [1] * 2)
]

print("Solutions to particle at rest")
for ψ in solutions:
    print(ψ)
    left = I * ħ * diff(ψ, t)
    right = me * c ** 2 * β @ ψ
    print(f"{left == right=}\n")


print("Energy:")
for ψ in solutions:
    print(simplify(ψ.H @ me * c ** 2 * β @ ψ)[0])


# Moving Particle (5.3.2)
ρ, x = symbols("ρ x")
u = Function("u")
ψ = u(ρ) * exp(-I * ρ * x / ħ)

# (c * α * ρ + β * me * c**2) * ψ = I * ħ * ψ.diff(t)
ρx, ρy, ρz = symbols("ρ_x ρ_y ρ_z")
ρ = [ρx, ρy, ρz]
dot(σ, ρ)
(c * dot(σ, ρ) + β * me * c ** 2) * ψ

# Due to the block structure of the α matrices
# c * σ * ρ @ ψs + me * c**2 * ψl = I * ħ * ψl.diff(t)
# c * σ * ρ @ ψl - me * c**2 * ψs = I * ħ * ψs.diff(t)

ψ, ψl, ψs, ψ1, ψ2, ψ3, ψ4 = symbols("ψ ψ_l ψ_s ψ_1 ψ_2 ψ_3 ψ_4")
ψ = Matrix([ψl, ψs])
ψ = Matrix([ψ1, ψ2, ψ3, ψ4])
ψl = Matrix([ψ1, ψ2])
ψs = Matrix([ψ3, ψ4])

ρx, ρy, ρz = symbols("ρ_x ρ_y ρ_z")
ρ = [ρx, ρy, ρz]
ρ = np.array([ρx, ρy, ρz])

l = c * dot(σ, ρ)
print(l[0, :])
print(l[1, :])


s = σ
s1 = σ1
s2 = σ2
s3 = σ3
p = ρ
px = ρx
py = ρy
pz = ρz
