from typing import NamedTuple
import jax.numpy as jnp
from jax import random
from models.activations import heaviside

from models.connectivity import generate_input_weights, generate_recurrent_weights
from models.readout import ReadoutParams, ReadoutState

CONNECTIVITY = 1.0

"""
following the paper: Advancing spatio-temporal processing in spiking neural networks through adaptation
by Maximilian Baronig, Romain Ferrand, Silvester Sabathiel and Robert Legenstein

except "k" is replaced with "t"

u_hat[t] = alpha*u[t - 1] + (1 - alpha) (-w[t - 1] + I[t]) (6a)  
w[t] = beta*w[t - 1] + (1 - beta) (a*u[t] + b*S[t]) (6b)

and the reset equation:
u[t] = u_hat[t](1 - S[t]) + u_rest*S[t]
"""


class AdLIFParams(NamedTuple):
    # neuron params
    alpha: jnp.ndarray
    beta: jnp.ndarray
    u_rest: jnp.ndarray
    u_thresh: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    # weights
    W_in: jnp.ndarray
    W_in_mask: jnp.ndarray
    W_rec: jnp.ndarray
    W_rec_mask: jnp.ndarray


class AdLIFState(NamedTuple):
    u: jnp.ndarray
    w: jnp.ndarray
    spike: jnp.ndarray


def generate_adlif_params(
    key: jnp.ndarray,
    n_input_neurons: int,
    n_hidden_neurons: int,
    has_recurrent_connections=False,
):
    key_in, key_rec = random.split(key, 2)
    W_in, W_in_mask = generate_input_weights(key_in, n_input_neurons, n_hidden_neurons)

    if has_recurrent_connections:
        W_rec, W_rec_mask = generate_recurrent_weights(key_rec, n_hidden_neurons)
    else:
        W_rec = jnp.zeros((n_hidden_neurons, n_hidden_neurons))
        W_rec_mask = jnp.zeros((n_hidden_neurons, n_hidden_neurons))

    return AdLIFParams(
        alpha=jnp.full(n_hidden_neurons, 0.96),
        beta=jnp.full(n_hidden_neurons, 0.96),
        u_rest=jnp.full(n_hidden_neurons, 0.0),
        u_thresh=jnp.full(n_hidden_neurons, 1.0),
        a=jnp.full(n_hidden_neurons, 40.0),
        b=jnp.full(n_hidden_neurons, 0.0),
        W_in=W_in,
        W_in_mask=W_in_mask,
        W_rec=W_rec,
        W_rec_mask=W_rec_mask,
    )


def generate_adlif_state(key: jnp.ndarray, n_hidden_neurons: int):
    return AdLIFState(
        u=jnp.zeros(n_hidden_neurons, dtype=jnp.float32),
        w=jnp.zeros(n_hidden_neurons, dtype=jnp.float32),
        spike=jnp.zeros(n_hidden_neurons, dtype=jnp.float32),
    )


def step(params: AdLIFParams, state: AdLIFState, inputs: jnp.ndarray):
    masked_W_in = params.W_in * params.W_in_mask
    masked_W_rec = params.W_rec * params.W_rec_mask
    incoming_current = inputs @ masked_W_in + state.spike @ masked_W_rec

    u_hat = params.alpha * state.u + (1 - params.alpha) * (-state.w + incoming_current)
    spike = heaviside(u_hat - params.u_thresh)
    u = u_hat * (1 - spike) + params.u_rest * spike
    w = params.beta * state.w + (1 - params.beta) * (params.a * u + params.b * spike)

    return AdLIFState(u=u, w=w, spike=spike)
