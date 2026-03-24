# PINNS (JAX + Equinox)

Minimal PINN examples for:
- 1D steady Poisson
- 2D steady Poisson
- 1D unsteady heat
- 2D unsteady heat

## Setup virtual environment

From the project root:

```bash
cd simple-pinns
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run scripts

```bash
python 01_1d_steady_poisson_equinox.py
python 02_2d_steady_poisson_equinox.py
python 03_1d_unsteady_heat_equinox.py
python 04_2d_unsteady_heat_equinox.py
```