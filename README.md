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
- **Scalability**: Provided with 10 hydrocarbon example components currently described with ideal behavior.
- **Thermodynamics**: Utilizes the DIPPR thermodynamic system descriptions to match the thermodynamic database provided by Aspen Plus.

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

- **`equdist/`**: Main directory containing the core code for the Newton-Raphson algorithm.
- **`notebooks/`**: Jupyter notebooks for training and evaluation.
- **`README.md`**: Project documentation.


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request. For major changes, please discuss them with the repository owner first.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to adjust this README to better fit the specific needs and details of your project. Let me know if there is anything specific you'd like to add or modify!
