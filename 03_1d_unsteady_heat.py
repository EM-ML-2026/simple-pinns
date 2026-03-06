"""
1D unsteady heat equation with JAX and Equinox.

This script solves u_t - u_xx = 0 on [0, 1] x [0, T].
Boundary conditions: u(0, t) = u(1, t) = 0.
Initial condition: u(x, 0) = sin(pi x).

Analytical solution: u(x, t) = sin(pi x) * exp(-pi^2 t).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from scipy.stats import qmc
from typing import Tuple


# Configuration
N_COLLOCATION_POINTS = 50
N_BOUNDARY_POINTS = 10
N_INITIAL_POINTS = 10
LEARNING_RATE = 1e-3
N_OPTIMIZATION_EPOCHS = 5000
BC_LOSS_WEIGHT = 100.0
IC_LOSS_WEIGHT = 100.0
SEED = 42
T_FINAL = 0.2


def create_network_spacetime(key: jr.PRNGKey) -> eqx.nn.MLP:
    """
    Create a space-time PINN network.

    Input: [x, t] in [0,1] x [0,T]
    Output: scalar u(x,t)
    Architecture: 2 -> 32 -> 32 -> 32 -> 32 -> 1
    """
    return eqx.nn.MLP(
        in_size=2,
        out_size=1,
        width_size=32,
        depth=4,
        activation=jnp.tanh,
        key=key,
    )


def initial_condition(x: jnp.ndarray) -> jnp.ndarray:
    """
    Initial condition: u(x,0) = sin(pi x)
    """
    return jnp.sin(jnp.pi * x)


def analytical_solution(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical solution: u(x,t) = sin(pi x) * exp(-pi^2 t)
    """
    return jnp.sin(jnp.pi * x) * jnp.exp(-(jnp.pi**2) * t)


def pde_residual(network: eqx.Module, xt: jnp.ndarray) -> jnp.ndarray:
    """
    Compute PDE residual: du/dt - d^2u/dx^2
    """
    def u_func(z):
        return jnp.squeeze(network(z))

    grad_u = jax.grad(u_func)(xt)
    u_t = grad_u[1]

    hessian_u = jax.hessian(u_func)(xt)
    u_xx = hessian_u[0, 0]

    return u_t - u_xx


def loss_fn(
    network: eqx.Module,
    xt_collocation: jnp.ndarray,
    xt_boundary: jnp.ndarray,
    x_initial: jnp.ndarray,
    u_initial: jnp.ndarray,
):
    """
    Total loss = PDE loss + BC loss + IC loss
    """
    residuals = jax.vmap(lambda xt: pde_residual(network, xt))(xt_collocation)
    pde_loss = 0.5 * jnp.mean(residuals**2)

    bc_predictions = jax.vmap(lambda xt: jnp.squeeze(network(xt)))(xt_boundary)
    bc_loss = 0.5 * jnp.mean(bc_predictions**2)

    xt_initial = jnp.stack([x_initial, jnp.zeros_like(x_initial)], axis=1)
    ic_predictions = jax.vmap(lambda xt: jnp.squeeze(network(xt)))(xt_initial)
    ic_loss = 0.5 * jnp.mean((ic_predictions - u_initial) ** 2)

    total_loss = pde_loss + BC_LOSS_WEIGHT * bc_loss + IC_LOSS_WEIGHT * ic_loss

    return total_loss, (pde_loss, bc_loss, ic_loss)


def generate_collocation_points(
    n_points: int,
    t_final: float,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """
    Generate interior collocation points in [0,1] x [0,T] using Latin Hypercube Sampling.
    Boundary and initial line are excluded.
    """
    sampler = qmc.LatinHypercube(d=2, seed=int(key[0]))
    samples = sampler.random(n_points)

    x_samples = samples[:, 0] * 0.98 + 0.01
    t_samples = samples[:, 1] * (t_final - 0.01) + 0.01

    return jnp.stack([x_samples, t_samples], axis=1)


def generate_boundary_points(n_points: int, t_final: float) -> jnp.ndarray:
    """
    Generate boundary points on x = 0 and x = 1 for t in [0,T].
    """
    n_half = n_points // 2
    t_boundary = jnp.linspace(0.0, t_final, n_half)

    left = jnp.stack([jnp.zeros_like(t_boundary), t_boundary], axis=1)
    right = jnp.stack([jnp.ones_like(t_boundary), t_boundary], axis=1)

    return jnp.vstack([left, right])


def generate_initial_points(n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate points on the initial line t = 0 with their target values.
    """
    x = jnp.linspace(0.0, 1.0, n_points)
    u = initial_condition(x)
    return x, u


def train_pinn(
    network: eqx.Module,
    xt_collocation: jnp.ndarray,
    xt_boundary: jnp.ndarray,
    x_initial: jnp.ndarray,
    u_initial: jnp.ndarray,
    n_epochs: int = N_OPTIMIZATION_EPOCHS,
):
    """
    Train the PINN using Adam optimizer.
    """
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    loss_history = []
    pde_loss_history = []
    bc_loss_history = []
    ic_loss_history = []

    @eqx.filter_jit
    def make_step(network, opt_state):
        (loss, (pde_loss, bc_loss, ic_loss)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(network, xt_collocation, xt_boundary, x_initial, u_initial)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(network, eqx.is_array)
        )
        new_network = eqx.apply_updates(network, updates)

        return new_network, new_opt_state, loss, pde_loss, bc_loss, ic_loss

    for epoch in range(n_epochs):
        network, opt_state, loss, pde_loss, bc_loss, ic_loss = make_step(
            network, opt_state
        )

        loss_history.append(float(loss))
        pde_loss_history.append(float(pde_loss))
        bc_loss_history.append(float(bc_loss))
        ic_loss_history.append(float(ic_loss))

        if epoch % 500 == 0:
            print(
                f"epoch {epoch:5d} "
                f"total {float(loss):.6e} "
                f"pde {float(pde_loss):.6e} "
                f"bc {float(bc_loss):.6e} "
                f"ic {float(ic_loss):.6e}"
            )

    return network, loss_history, pde_loss_history, bc_loss_history, ic_loss_history


def main():
    """
    Main training and evaluation script.
    """
    print("=" * 70)
    print("1D Unsteady Heat Equation with PINNs (Equinox)")
    print("=" * 70)

    key = jr.PRNGKey(SEED)
    key, init_key, coll_key = jr.split(key, 3)

    network = create_network_spacetime(init_key)
    print("\nNetwork created")
    print("Architecture: 2 [x,t] -> 32 -> 32 -> 32 -> 32 -> 1")
    print("Equation: du/dt - d^2u/dx^2 = 0")
    print("IC: u(x,0) = sin(pi x)")
    print("BC: u(0,t) = u(1,t) = 0")

    xt_collocation = generate_collocation_points(
        N_COLLOCATION_POINTS, T_FINAL, coll_key
    )
    xt_boundary = generate_boundary_points(N_BOUNDARY_POINTS, T_FINAL)
    x_initial, u_initial = generate_initial_points(N_INITIAL_POINTS)

    print("\nTraining data generated")
    print(f"Collocation points: {xt_collocation.shape[0]}")
    print(f"Boundary points:    {xt_boundary.shape[0]}")
    print(f"Initial points:     {x_initial.shape[0]}")

    print("\nTraining")
    print("-" * 70)
    network, loss_hist, pde_hist, bc_hist, ic_hist = train_pinn(
        network,
        xt_collocation,
        xt_boundary,
        x_initial,
        u_initial,
        N_OPTIMIZATION_EPOCHS,
    )

    x_test = jnp.linspace(0.0, 1.0, 50)
    t_test = jnp.linspace(0.0, T_FINAL, 30)
    X, T = jnp.meshgrid(x_test, t_test, indexing="xy")
    xt_test = jnp.stack([X.flatten(), T.flatten()], axis=1)

    u_pinn = jax.vmap(lambda xt: jnp.squeeze(network(xt)))(xt_test).reshape(X.shape)
    u_exact = jax.vmap(lambda xt: analytical_solution(xt[0], xt[1]))(xt_test).reshape(
        X.shape
    )

    error = jnp.abs(u_pinn - u_exact)
    max_error = jnp.max(error)
    mean_error = jnp.mean(error)
    l2_error = jnp.sqrt(jnp.mean(error**2))

    print("\nResults")
    print("-" * 70)
    print(f"Final total loss: {loss_hist[-1]:.6e}")
    print(f"Final PDE loss:   {pde_hist[-1]:.6e}")
    print(f"Final BC loss:    {bc_hist[-1]:.6e}")
    print(f"Final IC loss:    {ic_hist[-1]:.6e}")
    print(f"Max error:        {max_error:.6e}")
    print(f"Mean error:       {mean_error:.6e}")
    print(f"L2 error:         {l2_error:.6e}")
    print(f"Solution range:   [{jnp.min(u_pinn):.6f}, {jnp.max(u_pinn):.6f}]")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    cf = ax.contourf(X, T, u_pinn, levels=20)
    plt.colorbar(cf, ax=ax, label="u(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("PINN solution")

    ax = axes[0, 1]
    cf = ax.contourf(X, T, u_exact, levels=20)
    plt.colorbar(cf, ax=ax, label="u(x,t)")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("analytical solution")

    ax = axes[1, 0]
    cf = ax.contourf(X, T, error, levels=20)
    plt.colorbar(cf, ax=ax, label="|error|")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"absolute error (max: {float(max_error):.2e})")

    ax = axes[1, 1]
    ax.semilogy(loss_hist, label="total", linewidth=2)
    ax.semilogy(pde_hist, label="pde", alpha=0.7)
    ax.semilogy(bc_hist, label="bc", alpha=0.7)
    ax.semilogy(ic_hist, label="ic", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("03_1d_unsteady_heat.png", dpi=150, bbox_inches="tight")
    plt.show()

    return network, loss_hist


if __name__ == "__main__":
    network, history = main()