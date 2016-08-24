from setuptools import setup

config = {
    'description': 'Quantum Chemistry Code',
    'author': 'Jonathon Vandezande',
    'url': '',
    'install_requires': ['nose', 'sympy', 'numpy', 'matplotlib'],
    'tests_require': ['nose'],
    'name': 'quantum'
}

setup(**config)
