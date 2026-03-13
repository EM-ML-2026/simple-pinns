"""
2D steady Poisson equation with JAX and Equinox.

This script solves laplacian(u) + f(x, y) = 0 on [0, 1] x [0, 1].
Boundary condition: u(x, y) = exp(xy) on the domain boundary.

The manufactured solution is u(x, y) = exp(xy), which gives
laplacian(u) = exp(xy) * (x^2 + y^2), so
f(x, y) = -exp(xy) * (x^2 + y^2).
"""

import numpy as np
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
LEARNING_RATE = 1e-3
N_OPTIMIZATION_EPOCHS = 5000
BC_LOSS_WEIGHT = 100.0
SEED = 42


def create_network(key: jr.PRNGKey) -> eqx.nn.MLP:
    """
    Create a neural network approximating u(x,y).

    Input:
        [x, y]

    Output:
        scalar u(x,y)

    Architecture:
        2 -> 32 -> 32 -> 32 -> 32 -> 1
    """
    return eqx.nn.MLP(
        in_size=2,
        out_size=1,
        width_size=32,
        depth=4,
        activation=jnp.tanh,
        key=key,
    )


def analytical_solution(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical solution.

        u(x,y) = exp(x*y)
    """
    return jnp.exp(x * y)


def rhs_function(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Right-hand side forcing term.

    For the PDE

        laplacian(u) + f(x,y) = 0

    with exact solution u(x,y) = exp(x*y),

        laplacian(u) = exp(x*y) * (x^2 + y^2)

    therefore

        f(x,y) = -exp(x*y) * (x^2 + y^2)
    """
    return -jnp.exp(x * y) * (x**2 + y**2)


def pde_residual(network: eqx.Module, xy: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the PDE residual.

        residual = laplacian(u) + f(x,y)

    The Laplacian is computed using the Hessian.
    """
    def u_func(z):
        return jnp.squeeze(network(z))

    hessian = jax.hessian(u_func)(xy)
    laplacian = jnp.trace(hessian)

    x, y = xy

    return laplacian + rhs_function(x, y)


def loss_fn(
    network: eqx.Module,
    xy_collocation: jnp.ndarray,
    xy_boundary: jnp.ndarray,
    u_boundary: jnp.ndarray,
) -> jnp.ndarray:
    """
    Total loss = PDE loss + boundary condition loss.

    PDE loss:
        residual should be zero inside the domain.

    BC loss:
        network should match analytical boundary values.
    """

    residuals = jax.vmap(lambda xy: pde_residual(network, xy))(xy_collocation)
    pde_loss = 0.5 * jnp.mean(residuals**2)

    boundary_pred = jax.vmap(lambda xy: jnp.squeeze(network(xy)))(xy_boundary)
    bc_loss = 0.5 * jnp.mean((boundary_pred - u_boundary) ** 2)

    return pde_loss + BC_LOSS_WEIGHT * bc_loss


def bilinear_interp(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xi_x: jnp.ndarray,
    xi_y: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Bilinear interpolation of a 2D field at scalar point (x, y).

    xi_x : (Q_x,) x-coordinates of the uniform grid
    xi_y : (Q_y,) y-coordinates of the uniform grid
    values: (Q_x, Q_y) field values
    """
    dx = xi_x[1] - xi_x[0]
    dy = xi_y[1] - xi_y[0]
    cx = (x - xi_x[0]) / dx   # fractional grid coordinate in x
    cy = (y - xi_y[0]) / dy   # fractional grid coordinate in y
    return jax.scipy.ndimage.map_coordinates(
        values,
        [jnp.atleast_1d(cx), jnp.atleast_1d(cy)],
        order=1,
        mode='nearest',
    )[0]


def pde_residual_heterogeneous(
    network: eqx.Module,
    xy: jnp.ndarray,
    xi_x: jnp.ndarray,
    xi_y: jnp.ndarray,
    theta_grid: jnp.ndarray,
    f_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Residual for -div(theta * grad(u)) = f, expanded as:

        theta * laplacian(u) + grad(theta) . grad(u) + f = 0

    theta and f are bilinearly interpolated from the loaded grid data.
    """
    def u_func(z):
        return jnp.squeeze(network(z))

    hessian = jax.hessian(u_func)(xy)
    lap_u   = jnp.trace(hessian)
    grad_u  = jax.grad(u_func)(xy)

    def theta_at(z):
        return bilinear_interp(z[0], z[1], xi_x, xi_y, theta_grid)

    theta      = theta_at(xy)
    grad_theta = jax.grad(theta_at)(xy)   # differentiates through the interpolation

    f_val = bilinear_interp(xy[0], xy[1], xi_x, xi_y, f_grid)

    return theta * lap_u + jnp.dot(grad_theta, grad_u) + f_val


def loss_fn_heterogeneous(
    network: eqx.Module,
    xy_collocation: jnp.ndarray,
    xy_boundary: jnp.ndarray,
    u_boundary: jnp.ndarray,
    xi_x: jnp.ndarray,
    xi_y: jnp.ndarray,
    theta_grid: jnp.ndarray,
    f_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Total loss for the heterogeneous problem."""
    residuals = jax.vmap(
        lambda xy: pde_residual_heterogeneous(network, xy, xi_x, xi_y, theta_grid, f_grid)
    )(xy_collocation)
    pde_loss = 0.5 * jnp.mean(residuals**2)

    boundary_pred = jax.vmap(lambda xy: jnp.squeeze(network(xy)))(xy_boundary)
    bc_loss = 0.5 * jnp.mean((boundary_pred - u_boundary) ** 2)

    return pde_loss + BC_LOSS_WEIGHT * bc_loss


def generate_collocation_points(n_points: int, key: jr.PRNGKey) -> jnp.ndarray:
    """
    Generate interior collocation points using Latin Hypercube sampling.
    """

    sampler = qmc.LatinHypercube(d=2, seed=int(key[0]))
    samples = sampler.random(n_points)

    return jnp.array(samples * 0.98 + 0.01)


def generate_boundary_points(n_points_total: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate boundary points on the square domain.

    The boundary consists of four edges of the unit square.
    """

    n_per_side = n_points_total // 4
    t = jnp.linspace(0.0, 1.0, n_per_side)

    left = jnp.stack([jnp.zeros_like(t), t], axis=1)
    right = jnp.stack([jnp.ones_like(t), t], axis=1)
    bottom = jnp.stack([t, jnp.zeros_like(t)], axis=1)
    top = jnp.stack([t, jnp.ones_like(t)], axis=1)

    xy_boundary = jnp.vstack([left, right, bottom, top])

    u_boundary = jax.vmap(lambda xy: analytical_solution(xy[0], xy[1]))(xy_boundary)

    return xy_boundary, u_boundary


def train_pinn(
    network: eqx.Module,
    xy_collocation: jnp.ndarray,
    xy_boundary: jnp.ndarray,
    u_boundary: jnp.ndarray,
    n_epochs: int = N_OPTIMIZATION_EPOCHS,
):
    """
    Train the PINN using Adam optimizer.
    """

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    loss_history = []

    @eqx.filter_jit
    def step(network, opt_state):

        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            network, xy_collocation, xy_boundary, u_boundary
        )

        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(network, eqx.is_array)
        )

        network = eqx.apply_updates(network, updates)

        return network, opt_state, loss

    for epoch in range(n_epochs):

        network, opt_state, loss = step(network, opt_state)

        loss_history.append(float(loss))

        if epoch % 500 == 0:
            print("epoch", epoch, "loss", float(loss))

    return network, loss_history


def main():
    """
    Main training and evaluation script.
    """

    print("2D Poisson PINN example")

    key = jr.PRNGKey(SEED)
    key, init_key, coll_key = jr.split(key, 3)

    network = create_network(init_key)

    xy_collocation = generate_collocation_points(N_COLLOCATION_POINTS, coll_key)

    xy_boundary, u_boundary = generate_boundary_points(N_BOUNDARY_POINTS)

    network, loss_history = train_pinn(
        network,
        xy_collocation,
        xy_boundary,
        u_boundary,
        N_OPTIMIZATION_EPOCHS,
    )

    test_points = jnp.linspace(0.0, 1.0, 50)

    X, Y = jnp.meshgrid(test_points, test_points)

    xy_test = jnp.stack([X.flatten(), Y.flatten()], axis=1)

    u_pinn = jax.vmap(lambda xy: jnp.squeeze(network(xy)))(xy_test).reshape(X.shape)

    u_exact = jax.vmap(lambda xy: analytical_solution(xy[0], xy[1]))(
        xy_test
    ).reshape(X.shape)

    residual = jax.vmap(lambda xy: pde_residual(network, xy))(xy_test).reshape(X.shape)

    error = jnp.abs(u_pinn - u_exact)

    print("max error", float(jnp.max(error)))
    print("mean error", float(jnp.mean(error)))
    print("mean residual", float(jnp.mean(jnp.abs(residual))))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    cf = axes[0, 0].contourf(X, Y, u_pinn, levels=20)
    plt.colorbar(cf, ax=axes[0, 0])
    axes[0, 0].set_title("PINN solution")

    cf = axes[0, 1].contourf(X, Y, u_exact, levels=20)
    plt.colorbar(cf, ax=axes[0, 1])
    axes[0, 1].set_title("exact solution")

    cf = axes[1, 0].contourf(X, Y, error, levels=20)
    plt.colorbar(cf, ax=axes[1, 0])
    axes[1, 0].set_title("absolute error")

    axes[1, 1].semilogy(loss_history)
    axes[1, 1].set_title("training loss")

    plt.tight_layout()
    plt.savefig('02_2d_steady_poisson.png', dpi=150, bbox_inches='tight')
    print(f" Visualization saved to: 02_2d_steady_poisson.png")
    plt.show()

    return network

def main_heterogeneous():
    """
    Train and evaluate the PINN on a heterogeneous problem.
    Loads theta and f from the NGO-exported steadydiffusion_test_sample.npz.
    """
    print("2D Poisson PINN example (heterogeneous)")

    NPZ_PATH = '../NGO/examples/steadydiffusion_test_sample.npz'
    data = np.load(NPZ_PATH)
    xi_x       = jnp.array(data['xi_x'])    # (Q_x,)
    xi_y       = jnp.array(data['xi_y'])    # (Q_y,)
    theta_grid = jnp.array(data['theta'])   # (Q_x, Q_y)
    f_grid     = jnp.array(data['f'])       # (Q_x, Q_y)
    u_grid     = jnp.array(data['u'])       # (Q_x, Q_y) exact solution
    print(f"Loaded: theta in [{float(theta_grid.min()):.3f}, {float(theta_grid.max()):.3f}], "
          f"f in [{float(f_grid.min()):.3f}, {float(f_grid.max()):.3f}]")

    key = jr.PRNGKey(SEED)
    key, init_key, coll_key = jr.split(key, 3)

    network = create_network(init_key)
    xy_collocation = generate_collocation_points(N_COLLOCATION_POINTS, coll_key)

    # Boundary points; use interpolated exact solution for BCs
    xy_boundary, _ = generate_boundary_points(N_BOUNDARY_POINTS)
    u_boundary = jax.vmap(
        lambda xy: bilinear_interp(xy[0], xy[1], xi_x, xi_y, u_grid)
    )(xy_boundary)

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    loss_history = []

    @eqx.filter_jit
    def step(network, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn_heterogeneous)(
            network, xy_collocation, xy_boundary, u_boundary,
            xi_x, xi_y, theta_grid, f_grid,
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(network, eqx.is_array)
        )
        network = eqx.apply_updates(network, updates)
        return network, opt_state, loss

    for epoch in range(N_OPTIMIZATION_EPOCHS):
        network, opt_state, loss = step(network, opt_state)
        loss_history.append(float(loss))
        if epoch % 500 == 0:
            print("epoch", epoch, "loss", float(loss))

    # Evaluate on a 50x50 test grid
    test_points = jnp.linspace(0.0, 1.0, 50)
    Xg, Yg = jnp.meshgrid(test_points, test_points)
    xy_test = jnp.stack([Xg.flatten(), Yg.flatten()], axis=1)

    u_pinn  = jax.vmap(lambda xy: jnp.squeeze(network(xy)))(xy_test).reshape(Xg.shape)
    u_exact = jax.vmap(
        lambda xy: bilinear_interp(xy[0], xy[1], xi_x, xi_y, u_grid)
    )(xy_test).reshape(Xg.shape)
    error = jnp.abs(u_pinn - u_exact)

    print("max error",  float(jnp.max(error)))
    print("mean error", float(jnp.mean(error)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    cf = axes[0, 0].contourf(Xg, Yg, u_pinn, levels=20)
    plt.colorbar(cf, ax=axes[0, 0])
    axes[0, 0].set_title("PINN solution (heterogeneous)")

    cf = axes[0, 1].contourf(Xg, Yg, u_exact, levels=20)
    plt.colorbar(cf, ax=axes[0, 1])
    axes[0, 1].set_title("exact solution")

    cf = axes[1, 0].contourf(Xg, Yg, error, levels=20)
    plt.colorbar(cf, ax=axes[1, 0])
    axes[1, 0].set_title("absolute error")

    axes[1, 1].semilogy(loss_history)
    axes[1, 1].set_title("training loss")

    plt.tight_layout()
    plt.savefig('02_2d_steady_poisson_heterogeneous.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: 02_2d_steady_poisson_heterogeneous.png")
    plt.show()

    return network


if __name__ == "__main__":
    network = main_heterogeneous()