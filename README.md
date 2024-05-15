# GNN4CIRCUITS

GNN4CIRCUITS is a platform designed for applying Graph Neural Networks (GNNs) to circuit analysis and design. This repository includes the necessary files and instructions to set up and use the platform.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Git

### Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2. **Create the conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the environment:**

    ```bash
    conda activate hw2vec
    ```

4. **Verify the installation:**

    Ensure all the required libraries are installed correctly by running:

    ```bash
    conda list
    ```

## Usage

### Parsing and Analyzing Verilog Data

To parse and analyze Verilog files, use the `parse` command with the appropriate options:

```bash
python GNN4CIRCUITS.py parse -ver <path-to-verilog-file> -hw <hardware-type> [-lib <path-to-library>] [optional-parameters]
