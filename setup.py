import os
from typing import List

import setuptools
from setuptools import setup


setup(
    name="EquDist",
    version=0.0.1,
    author="NiklasSlager",
    author_email="niklasslager@outlook.com",
    description="Equilibrium separation model in JAX",
    license="Apache 2.0",
    url="https://github.com/NiklasSlager/EquDist/",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="Chemical separation JAX",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=["chex==0.1.86", "jax==0.4.28", "jaxlib==0.4.28", "jaxopt==0.8.3"]
    package_data={"EquDist": ["py.typed"]},
    classifiers=[  
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
    include_package_data=True,
)
