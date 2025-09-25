from typing import NamedTuple
import jax.numpy as jnp
from jax import random

CONNECTIVITY = 1.0


class ReadoutParams(NamedTuple):
    W_in: jnp.ndarray
    W_in_mask: jnp.ndarray


class ReadoutState(NamedTuple):
    accumulated_spikes: jnp.ndarray


def generate_readout_params(
    key: jnp.ndarray, n_input_neurons: int, n_output_neurons: int
):
    key_in, key_mask = random.split(key)
    limit_in = jnp.sqrt(6.0 / (n_input_neurons + n_output_neurons))
    W_in = random.uniform(
        key_in, (n_input_neurons, n_output_neurons), minval=-limit_in, maxval=limit_in
    )
    W_in_mask = random.bernoulli(
        key_mask, CONNECTIVITY, (n_input_neurons, n_output_neurons)
    ).astype(jnp.float32)
    return ReadoutParams(W_in=W_in, W_in_mask=W_in_mask)


def generate_readout_state(n_output_neurons: int):
    return ReadoutState(
        accumulated_spikes=jnp.zeros((1, n_output_neurons)),
    )


def readout_step(params: ReadoutParams, state: ReadoutState, spike: jnp.ndarray):
    masked_W_in = params.W_in * params.W_in_mask
    weighted_input = spike @ masked_W_in
    accumulated_spikes = state.accumulated_spikes + weighted_input
    return ReadoutState(accumulated_spikes=accumulated_spikes)
