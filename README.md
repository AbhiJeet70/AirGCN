# AirGCN

## Overview
AirGCN is a Graph Convolutional Network (GCN) implementation using the PyTorch Geometric library. This project aims to train and evaluate GCN models on the Airports dataset for different countries, including the USA, Brazil, and Europe.

## Repository Structure
The repository contains the following files and directories:

- `data/`: Directory to store the downloaded Airports datasets.
- `models/`: Directory containing the GCN model implementation.
- `utils/`: Directory containing utility functions for data loading, augmentation, and quality checks.
- `train.py`: Script to train and evaluate the GCN models.

## Installation
To use this repository, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/AirGCN.git
    cd AirGCN
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    - Python 3.7+
    - PyTorch
    - PyTorch Geometric
    - NumPy
    ```

## Usage
To train and evaluate the GCN models on the Airports dataset, run the `train.py` script:

```sh
python train.py
```

The script processes each country's dataset, prints dataset statistics, trains the model, and outputs the best accuracy.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
