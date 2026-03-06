"""
1D steady Poisson equation with JAX and Equinox.

This script solves u_xx = -f(x) on [0, 1] with u(0) = u(1) = 0.
It uses f(x) = pi^2 sin(pi x), so the analytical solution is u(x) = sin(pi x).

Reference:
https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/physics_informed_neural_networks/poisson_pinn_in_jax_equinox.ipynb
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import Tuple, Callable

# Configuration
N_COLLOCATION_POINTS = 50
LEARNING_RATE = 1e-3
N_OPTIMIZATION_EPOCHS = 5000
BC_LOSS_WEIGHT = 100.0
SEED = 42


def create_network(key: jr.PRNGKey) -> eqx.nn.MLP:
    """
    Create a simple MLP neural network.
    
    Input: scalar x in [0, 1]
    Output: scalar u(x)
    Architecture: scalar -> 10 -> 10 -> scalar (with sigmoid activation)
    """
    return eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=10,
        depth=2,  # Total 2 hidden layers
        activation=jax.nn.sigmoid,
        key=key,
    )


def rhs_function(x: jnp.ndarray) -> jnp.ndarray:
    """
    Right-hand side of the PDE: f(x) = pi^2 sin(pi x)
    
    Analytical solution: u(x) = sin(pi x)
    Satisfies: u(0) = 0, u(1) = 0, u''(x) = -pi^2 sin(pi x) = -f(x)
    """
    return jnp.pi**2 * jnp.sin(jnp.pi * x)


def pde_residual(network: eqx.Module, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute PDE residual: d^2u/dx^2 + f(x)
    Uses automatic differentiation (twice) to compute second derivative.
    """
    # Wrap scalar x into 1D array for network input, squeeze output to scalar
    def u_func(x_scalar):
        return jnp.squeeze(network(jnp.array([x_scalar])))

    # Second derivative    
    u_double_prime = jax.grad(lambda x: jax.grad(u_func)(x))(x)
    # Residual
    return u_double_prime + rhs_function(x)


def loss_fn(network: eqx.Module, 
            collocation_points: jnp.ndarray) -> jnp.ndarray:
    """
    Total loss = PDE loss + BC loss (with weighting)
    """
    # PDE loss: residual should be around 0 at collocation points
    residuals = jax.vmap(lambda x: pde_residual(network, x))(collocation_points)
    pde_loss = 0.5 * jnp.mean(jnp.square(residuals))
    
    # Boundary condition loss: u(0) = 0 and u(1) = 0
    left_bc = jnp.squeeze(network(jnp.array([0.0]))) - 0.0
    right_bc = jnp.squeeze(network(jnp.array([1.0]))) - 0.0
    bc_loss = 0.5 * jnp.square(left_bc) + 0.5 * jnp.square(right_bc)
    
    # Total weighted loss
    total_loss = pde_loss + BC_LOSS_WEIGHT * bc_loss
    return total_loss


def analytical_solution(x: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical solution: u(x) = sin(pi x)
    """
    return jnp.sin(jnp.pi * x)


def train_pinn(network: eqx.Module,
               collocation_points: jnp.ndarray,
               n_epochs: int = N_OPTIMIZATION_EPOCHS) -> Tuple[eqx.Module, list]:
    """
    Train the PINN using Adam optimizer with JIT compilation.
    """
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    loss_history = []
    
    # JIT-compiled training step
    @eqx.filter_jit
    def make_step(network, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(network, collocation_points)
        updates, new_opt_state = optimizer.update(grads, opt_state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_opt_state, loss
    
    # Training loop
    for epoch in range(n_epochs):
        network, opt_state, loss = make_step(network, opt_state)
        loss_history.append(float(loss))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d}, Loss: {loss:.6f}")
    
    return network, loss_history


def main():
    """Main training and evaluation script."""
    print("=" * 60)
    print("1D Steady Poisson Equation with PINNs (Equinox)")
    print("=" * 60)
    
    # Initialize
    key = jr.PRNGKey(SEED)
    key, init_key, sample_key = jr.split(key, 3)
    
    # Create network
    network = create_network(init_key)
    print(f"\n Network created")
    print(f"  Architecture: 1D -> 10 -> 10 -> 1D (sigmoid)")
    
    # Generate collocation points (interior domain)
    collocation_points = jr.uniform(
        sample_key,
        (N_COLLOCATION_POINTS,),
        minval=0.001,
        maxval=0.999
    )
    print(f" {N_COLLOCATION_POINTS} collocation points generated")
    
    # Train
    print(f"\n{'Training' * 1}")
    print("-" * 60)
    network, loss_history = train_pinn(network, collocation_points, N_OPTIMIZATION_EPOCHS)
    print(f" Training completed")
    
    # Evaluate on test mesh
    test_mesh = jnp.linspace(0.0, 1.0, 200)
    pinn_solution = jax.vmap(network)(test_mesh.reshape(-1, 1)).flatten()
    
    # Get analytical solution
    analytical = analytical_solution(test_mesh)
    
    # Compute error
    error = jnp.abs(pinn_solution - analytical)
    max_error = jnp.max(error)
    mean_error = jnp.mean(error)
    l2_error = jnp.sqrt(jnp.mean(error**2))
    
    print(f"\n{'Results' * 1}")
    print("-" * 60)
    print(f"Max Error:  {max_error:.6e}")
    print(f"Mean Error: {mean_error:.6e}")
    print(f"L2 Error:   {l2_error:.6e}")
    print(f"Final Loss: {loss_history[-1]:.6e}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Solution comparison
    axes[0].plot(test_mesh, analytical, 'b-', label='Analytical Solution', linewidth=2)
    axes[0].plot(test_mesh, pinn_solution, 'r--', label='PINN Solution', alpha=0.7)
    axes[0].scatter(collocation_points, jnp.zeros_like(collocation_points), 
                   alpha=0.3, s=10, label='Collocation Points')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('Solution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Error
    axes[1].semilogy(test_mesh, error)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title(f'Absolute Error (max: {max_error:.2e})')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Loss convergence
    axes[2].semilogy(loss_history)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Convergence')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_1d_steady_poisson.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n Visualization saved to: 01_1d_steady_poisson.png")
    
    return network, loss_history


if __name__ == "__main__":
    network, history = main()
    print("\n Complete!")
