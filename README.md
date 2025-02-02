# Lazy and Feature training in artificial neural networks

M2 Project - Theoretical Principles of Deep Learning

Author: R. Bordas
Date: February 2025

This repository gathers all numerical experiments performed in this project (including some not in the report for space constraints).

## Code organization:

- Main model: FC network. 

- Main script (entry point): `run_experiments.py` (change the path of the Fashion MNIST data).
By default, download is disabled (can be activited in `datasets.py`).

- Results are saved in a save, change the path at the end of `run_experiments.py`

- `figure_2.py` is dependent on the ouputs of `run_experiments.py`
