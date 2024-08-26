# equdist

Welcome to the `equdist` repository! This project contains the code and resources necessary for implementing a Python-integrated JAX-written rigorous Newton-Raphson algorithm for equilibrium separation, with applications to distillation sequencing for purifying multi-component mixtures. 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to provide an efficient and reliable method for distillation sequencing using a Newton-Raphson algorithm implemented in JAX. The primary application is to leverage reinforcement learning to optimize distillation processes, providing a scalable and flexible solution for chemical engineering tasks.

## Features

- **Rigorous Newton-Raphson Algorithm**: Implemented in JAX for fast and reliable equilibrium separation.
- **Automatic Differentiation**: Supports automatic differentiation for efficient Jacobian calculation.
- **Scalability**: Tested with up to ten hydrocarbon components exhibiting ideal behavior, with potential for expansion to more complex mixtures.
- **Thermodynamics**: Leverages the DIPPR thermodynamic system and is compatible with the Aspen Plus thermodynamic database, making it adaptable to various chemical processes.

## Installation

To run this project, the repository can be installed via;

    ```bash
    pip install git+https://github.com/NiklasSlager/equdist.git
    ```

## Usage

1. **Run the Training Notebook**:
    A Jupyter notebook named `tutorial.ipynb` is provided to run the model. A link to Google Colab is provide for easy testing and use.

2. **Distillation input Parameters**:
   The input parameters to the distillation column are:
   1. Number of stages
   2. Feed location
   3. Feed rate (kmol/hr)
   4. Feed composition
   5. Reflux ratio
   6. Distillate rate
   7. Operating pressure

4. **Run the Algorithm**:
    The plots visualize profiles including liquid composition, liquid and vapor flowrate, temperature, and liquid and vapor enthalpies.

## Project Structure
The algorithm entails two steps for better convergence performance: In the first step, the equilibrium separation is solved assuming equimolar overflow. The heat equation is replaced with a total flow constraint to solve the Newton-Raphson procedure assuming equimolar overflow. The full set of MESH equations is solved in the second step.
- **`equdist/`**: Main directory containing the core code for the Newton-Raphson algorithm.
  - **`equimolar`**: Module for the equimolar overflow procedure.
  - **`model`**: Module for the full NR procedure.
  - **`functions`** auxilary functions used within the algorithm.
  - **`thermodynamics`** Description of the thermodynamic model using the DIPPR equations
  - **`costing`** Module for a basic total annualized cost estimation procedure based on the Marshall & Swift correlations
- **`notebooks/`**: Jupyter notebooks for example usage.
- **`README.md`**: Project documentation.


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request. For major changes, please discuss them with the repository owner first.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



