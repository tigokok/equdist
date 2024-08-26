import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="equdist",
    version="0.0.1",
    author="Niklas Slager",
    author_email="niklasslager@outlook.com",
    description="JAX implementation of Napthali-Sandholm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NiklasSlager/EquDist",
    packages=['equdist'],
    package_data={'eqdist': ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["chex==0.1.86", "jax==0.4.28", "jaxlib==0.4.28", "jaxopt==0.8.3"]
)

