import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JAX distillation",
    version="0.0.1",
    author="Niklas Slager",
    author_email="niklasslager@outlook.com",
    description="JAX implementation of Napthali-Sandholm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NiklasSlager/EquDist",
    packages=['NR_model_test'],
    package_data={'NR_model_test': [
        'Pure component parameters/Antoine.csv', 'Pure component parameters/Vapor_CP.csv',
        'Pure component parameters/Heat_of_evaporization.csv', 'Pure component parameters/Heat_of_formation.csv', 
    ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy==1.18.1","scipy == 1.10.0"]
)
install_requires=["chex==0.1.86", "jax==0.4.28", "jaxlib==0.4.28", "jaxopt==0.8.3"]
