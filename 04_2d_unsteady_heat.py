"""
2D unsteady heat equation with JAX and Equinox.

This script solves u_t - (u_xx + u_yy) = 0 on [0, 1] x [0, 1] x [0, T].
Boundary condition: u = 0 on the spatial boundary.
Initial condition: u(x, y, 0) = sin(pi x) * sin(pi y).

Analytical solution: u(x, y, t) = sin(pi x) * sin(pi y) * exp(-2 pi^2 t).
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
N_COLLOCATION_POINTS = 1000
N_BOUNDARY_POINTS = 120
N_INITIAL_POINTS = 120
LEARNING_RATE = 1e-3
N_OPTIMIZATION_EPOCHS = 5000
BC_LOSS_WEIGHT = 100.0
IC_LOSS_WEIGHT = 100.0
SEED = 42
T_FINAL = 0.1


def create_network_spacetime(key: jr.PRNGKey) -> eqx.nn.MLP:
    """
    Create a 2D space-time PINN network.

    Input: [x, y, t] in [0,1] x [0,1] x [0,T]
    Output: scalar u(x,y,t)
    Architecture: 3 -> 48 -> 48 -> 48 -> 48 -> 1
    """
    return eqx.nn.MLP(
        in_size=3,
        out_size=1,
        width_size=48,
        depth=4,
        activation=jnp.tanh,
        key=key,
    )


def initial_condition(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Initial condition: u(x,y,0) = sin(pi x) * sin(pi y)
    """
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def analytical_solution(
    x: jnp.ndarray,
    y: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Analytical solution: u(x,y,t) = sin(pi x) * sin(pi y) * exp(-2 pi^2 t)
    """
    return (
        jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        * jnp.exp(-2.0 * (jnp.pi**2) * t)
    )


def pde_residual(network: eqx.Module, xyt: jnp.ndarray) -> jnp.ndarray:
    """
    Compute PDE residual: du/dt - d^2u/dx^2 - d^2u/dy^2
    """
    def u_func(z):
        return jnp.squeeze(network(z))

    grad_u = jax.grad(u_func)(xyt)
    u_t = grad_u[2]

    hessian_u = jax.hessian(u_func)(xyt)
    u_xx = hessian_u[0, 0]
    u_yy = hessian_u[1, 1]

    return u_t - u_xx - u_yy


def loss_fn(
    network: eqx.Module,
    xyt_collocation: jnp.ndarray,
    xyt_boundary: jnp.ndarray,
    xy_initial: jnp.ndarray,
    u_initial: jnp.ndarray,
):
    """
    Total loss = PDE loss + BC loss + IC loss
    """
    residuals = jax.vmap(lambda xyt: pde_residual(network, xyt))(xyt_collocation)
    pde_loss = 0.5 * jnp.mean(residuals**2)

    bc_predictions = jax.vmap(lambda xyt: jnp.squeeze(network(xyt)))(xyt_boundary)
    bc_loss = 0.5 * jnp.mean(bc_predictions**2)

    xyt_initial = jnp.concatenate(
        [xy_initial, jnp.zeros((xy_initial.shape[0], 1))],
        axis=1,
    )
    ic_predictions = jax.vmap(lambda xyt: jnp.squeeze(network(xyt)))(xyt_initial)
    ic_loss = 0.5 * jnp.mean((ic_predictions - u_initial) ** 2)

    total_loss = pde_loss + BC_LOSS_WEIGHT * bc_loss + IC_LOSS_WEIGHT * ic_loss

    return total_loss, (pde_loss, bc_loss, ic_loss)


def generate_collocation_points(
    n_points: int,
    t_final: float,
    key: jr.PRNGKey,
) -> jnp.ndarray:
    """
    Generate interior collocation points in [0,1] x [0,1] x [0,T]
    using Latin Hypercube Sampling.

    Spatial boundaries and t = 0 are excluded.
    """
    sampler = qmc.LatinHypercube(d=3, seed=int(key[0]))
    samples = sampler.random(n_points)

    x_samples = samples[:, 0] * 0.98 + 0.01
    y_samples = samples[:, 1] * 0.98 + 0.01
    t_samples = samples[:, 2] * (t_final - 0.01) + 0.01

    return jnp.stack([x_samples, y_samples, t_samples], axis=1)


def generate_boundary_points(n_points: int, t_final: float) -> jnp.ndarray:
    """
    Generate boundary points on the four spatial faces:

        x = 0
        x = 1
        y = 0
        y = 1

    Each face is sampled over its free spatial coordinate and time.
    """
    n_per_face = n_points // 4
    n_side = int(jnp.sqrt(n_per_face))

    s = jnp.linspace(0.0, 1.0, n_side)
    t = jnp.linspace(0.0, t_final, n_side)
    S, T = jnp.meshgrid(s, t, indexing="xy")

    s_flat = S.flatten()
    t_flat = T.flatten()

    x0 = jnp.stack([jnp.zeros_like(s_flat), s_flat, t_flat], axis=1)
    x1 = jnp.stack([jnp.ones_like(s_flat), s_flat, t_flat], axis=1)
    y0 = jnp.stack([s_flat, jnp.zeros_like(s_flat), t_flat], axis=1)
    y1 = jnp.stack([s_flat, jnp.ones_like(s_flat), t_flat], axis=1)

    return jnp.vstack([x0, x1, y0, y1])


def generate_initial_points(n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate points on the initial slice t = 0 and their target values.
    """
    n_per_dim = int(jnp.sqrt(n_points))

    x = jnp.linspace(0.0, 1.0, n_per_dim)
    y = jnp.linspace(0.0, 1.0, n_per_dim)
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    xy = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    u = initial_condition(xy[:, 0], xy[:, 1])

    return xy, u


def train_pinn(
    network: eqx.Module,
    xyt_collocation: jnp.ndarray,
    xyt_boundary: jnp.ndarray,
    xy_initial: jnp.ndarray,
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
            loss_fn,
            has_aux=True,
        )(network, xyt_collocation, xyt_boundary, xy_initial, u_initial)

        updates, new_opt_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(network, eqx.is_array),
        )
        new_network = eqx.apply_updates(network, updates)

        return new_network, new_opt_state, loss, pde_loss, bc_loss, ic_loss

    for epoch in range(n_epochs):
        network, opt_state, loss, pde_loss, bc_loss, ic_loss = make_step(
            network,
            opt_state,
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
    print("2D Unsteady Heat Equation with PINNs (Equinox)")
    print("=" * 70)

    key = jr.PRNGKey(SEED)
    key, init_key, coll_key = jr.split(key, 3)

    network = create_network_spacetime(init_key)

    print("\nNetwork created")
    print("Architecture: 3 [x,y,t] -> 48 -> 48 -> 48 -> 48 -> 1")
    print("Equation: du/dt - d^2u/dx^2 - d^2u/dy^2 = 0")
    print("IC: u(x,y,0) = sin(pi x) * sin(pi y)")
    print("BC: u = 0 on the spatial boundary")

    xyt_collocation = generate_collocation_points(
        N_COLLOCATION_POINTS,
        T_FINAL,
        coll_key,
    )
    xyt_boundary = generate_boundary_points(N_BOUNDARY_POINTS, T_FINAL)
    xy_initial, u_initial = generate_initial_points(N_INITIAL_POINTS)

    print("\nTraining data generated")
    print(f"Collocation points: {xyt_collocation.shape[0]}")
    print(f"Boundary points:    {xyt_boundary.shape[0]}")
    print(f"Initial points:     {xy_initial.shape[0]}")

    print("\nTraining")
    print("-" * 70)
    network, loss_hist, pde_hist, bc_hist, ic_hist = train_pinn(
        network,
        xyt_collocation,
        xyt_boundary,
        xy_initial,
        u_initial,
        N_OPTIMIZATION_EPOCHS,
    )

    print("\nResults")
    print("-" * 70)
    print(f"Final total loss: {loss_hist[-1]:.6e}")
    print(f"Final PDE loss:   {pde_hist[-1]:.6e}")
    print(f"Final BC loss:    {bc_hist[-1]:.6e}")
    print(f"Final IC loss:    {ic_hist[-1]:.6e}")

    x_test = jnp.linspace(0.0, 1.0, 30)
    y_test = jnp.linspace(0.0, 1.0, 30)

    X_final, Y_final = jnp.meshgrid(x_test, y_test, indexing="xy")
    xyt_final = jnp.stack(
        [
            X_final.flatten(),
            Y_final.flatten(),
            jnp.full_like(X_final.flatten(), T_FINAL),
        ],
        axis=1,
    )

    u_pinn_final = jax.vmap(lambda xyt: jnp.squeeze(network(xyt)))(xyt_final).reshape(
        X_final.shape
    )
    u_exact_final = jax.vmap(
        lambda xyt: analytical_solution(xyt[0], xyt[1], xyt[2])
    )(xyt_final).reshape(X_final.shape)

    error_final = jnp.abs(u_pinn_final - u_exact_final)

    print(f"Max error at t={T_FINAL}:  {float(jnp.max(error_final)):.6e}")
    print(f"Mean error at t={T_FINAL}: {float(jnp.mean(error_final)):.6e}")
    print(f"L2 error at t={T_FINAL}:   {float(jnp.sqrt(jnp.mean(error_final**2))):.6e}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    cf = ax.contourf(X_final, Y_final, u_pinn_final, levels=20)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"PINN solution at t={T_FINAL}")

    ax = axes[0, 1]
    cf = ax.contourf(X_final, Y_final, u_exact_final, levels=20,)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"analytical solution at t={T_FINAL}")

    ax = axes[1, 0]
    cf = ax.contourf(X_final, Y_final, error_final, levels=20)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"absolute error (max: {float(jnp.max(error_final)):.2e})")

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
    plt.savefig("04_2d_unsteady_heat.png", dpi=150, bbox_inches="tight")
    plt.show()

    return network, loss_hist


if __name__ == "__main__":
    network, history = main()