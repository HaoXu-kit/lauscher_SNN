from typing import NamedTuple
import jax.numpy as jnp
from jax import random
from models.activations import heaviside

from models.connectivity import generate_input_weights, generate_recurrent_weights


class LIFParams(NamedTuple):
    alpha: jnp.ndarray
    v_thresh: jnp.ndarray
    W_in: jnp.ndarray
    W_in_mask: jnp.ndarray
    W_rec: jnp.ndarray
    W_rec_mask: jnp.ndarray


class LIFState(NamedTuple):
    v: jnp.ndarray
    spike: jnp.ndarray


def generate_lif_params(
    key: jnp.ndarray,
    n_input_neurons: int,
    n_hidden_neurons: int,
    has_recurrent_connections=False,
    v_thresh: float = 1.0,
):
    key_in, key_rec = random.split(key, 2)
    W_in, W_in_mask = generate_input_weights(key_in, n_input_neurons, n_hidden_neurons)

    if has_recurrent_connections:
        W_rec, W_rec_mask = generate_recurrent_weights(key_rec, n_hidden_neurons)
    else:
        W_rec = jnp.zeros((n_hidden_neurons, n_hidden_neurons))
        W_rec_mask = jnp.zeros((n_hidden_neurons, n_hidden_neurons))

    return LIFParams(
        alpha=jnp.full(n_hidden_neurons, 0.96),
        v_thresh=jnp.full(n_hidden_neurons, v_thresh),
        W_in=W_in,
        W_in_mask=W_in_mask,
        W_rec=W_rec,
        W_rec_mask=W_rec_mask,
    )


def generate_lif_state(key: jnp.ndarray, n_hidden_neurons: int):
    return LIFState(
        v=jnp.zeros((1, n_hidden_neurons)),
        spike=jnp.zeros((1, n_hidden_neurons), dtype=jnp.float32),
    )


def lif_step(params: LIFParams, state: LIFState, inputs: jnp.ndarray):
    masked_W_in = params.W_in * params.W_in_mask
    masked_W_rec = params.W_rec * params.W_rec_mask
    incoming_current = inputs @ masked_W_in + state.spike @ masked_W_rec
    v_hat = params.alpha * state.v + incoming_current
    spike = heaviside(v_hat - params.v_thresh)
    v = v_hat * (1 - spike)

    return LIFState(v=v, spike=spike)
