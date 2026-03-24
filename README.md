# PINNS (JAX + Equinox)

Minimal PINN examples for:
- 1D steady Poisson
- 2D steady Poisson
- 1D unsteady heat
- 2D unsteady heat

## Setup virtual environment

From the project root:

```bash
cd simple_pinns
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## GPU note

requirements.txt includes CUDA-enabled JAX (jax[cuda12]).
If an NVIDIA GPU and compatible driver are available, JAX will use GPU automatically.
Otherwise, it falls back to CPU.

Check backend:

```bash
python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

## Run scripts

```bash
python 01_1d_steady_poisson_equinox.py
python 02_2d_steady_poisson_equinox.py
python 03_1d_unsteady_heat_equinox.py
python 04_2d_unsteady_heat_equinox.py
```