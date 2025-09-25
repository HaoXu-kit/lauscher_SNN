import jax.numpy as jnp
from jax import custom_vjp


@custom_vjp
def heaviside(x):
    # Primal function
    return jnp.heaviside(x, 0.0)


def heaviside_forward(x_input):
    primal_output = heaviside(x_input)
    # Save the original x_input as a residual for the backward pass
    residuals = (x_input,)
    return primal_output, residuals


def heaviside_backward(residuals, g):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """
    (x_input_saved,) = residuals
    # The gradient is passed through only in a small window around the threshold (-0.5 < x <= 0.5)
    surrogate_grad_mask = (x_input_saved > -0.5) * (x_input_saved <= 0.5)
    return (g * surrogate_grad_mask,)


heaviside.defvjp(heaviside_forward, heaviside_backward)
