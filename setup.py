from setuptools import find_packages, setup

setup(
    name="FourDimensionalData",
    version="0.1",
    license="GPLv3",
    author="harripj",
    author_email="harrison.p.j@icloud.com",
    description="Readers for 4-dimensional data files.",
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages("."),
    install_requires=[
        "hyperspy",
        "ipywidgets",
        "KED",
        "matplotlib >= 3.4",
        "numpy",
        "pyqtgraph",
        "PySide2",
        "scipy",
        "tqdm",
    ],
    package_data={"": ["LICENSE", "README.md"]},
)
