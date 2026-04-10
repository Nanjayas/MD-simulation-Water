from setuptools import setup, find_packages

setup(
    name="mdwater",
    version="0.1.0",
    description="A Molecular Dynamics engine for simulating water using the Hamiltonian energy functional for a flexible SPC water model.",
    author="Yajie Zhang, Ningxin Wang, Nandana Jaya Sunil Kumar, Ayday Iskenderova, Itzel Jessica Martinez Marcelo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pytest>=7.0.0"
    ],
    python_requires=">=3.8",
)