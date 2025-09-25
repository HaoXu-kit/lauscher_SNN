from typing import NamedTuple
from jax.experimental import sparse
from jax import numpy as jnp
from jax import random

from models.activations import heaviside

# data = jnp.array([1.0, 2.0, 3.0])
# indices = jnp.array([[0, 0], [1, 1], [2, 2]])  # Diagonal elements
# shape = (3, 3)
# bcoo = sparse.BCOO((data, indices), shape=shape)


class LIFParams(NamedTuple):
    alpha: jnp.ndarray
    v_thresh: jnp.ndarray
    W_in: sparse.BCOO
    W_rec: sparse.BCOO


class LIFState(NamedTuple):
    v: jnp.ndarray
    spike: sparse.BCOO


class ReadoutParams(NamedTuple):
    W_in: sparse.BCOO


class ReadoutState(NamedTuple):
    accumulated_spikes: jnp.ndarray


class NetworkParams(NamedTuple):
    hidden_params: list[LIFParams]
    output_params: ReadoutParams


class NetworkState(NamedTuple):
    hidden_states: list[LIFState]
    output_state: ReadoutState


def generate_lif_params(
    key: jnp.ndarray,
    n_input_neurons: int,
    n_hidden_neurons: int,
    has_recurrent_connections=False,
    initial_decay=0.9,
):
    # TODO implement

    key_in, key_rec = random.split(key)
    # TODO pass in generator for range of values
    # https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.random_bcoo.html#jax.experimental.sparse.random_bcoo
    W_in = sparse.random_bcoo(key, (n_input_neurons, n_hidden_neurons))

    if has_recurrent_connections:
        W_rec = sparse.random_bcoo(key, (n_hidden_neurons, n_hidden_neurons))
    else:
        W_rec = sparse.empty((n_hidden_neurons, n_hidden_neurons))

    return LIFParams(
        alpha=jnp.ones(n_hidden_neurons) * initial_decay,
        v_thresh=jnp.ones(n_hidden_neurons),
        W_in=W_in,
        W_rec=W_rec,
    )


def generate_readout_params(
    key: jnp.ndarray, n_input_neurons: int, n_output_neurons: int
):
    # TODO implement
    W_in = sparse.random_bcoo(key, (n_input_neurons, n_output_neurons))
    return ReadoutParams(W_in=W_in)


def generate_lif_state(key: jnp.ndarray, n_hidden_neurons: int):
    return LIFState(
        v=jnp.zeros(n_hidden_neurons), spike=sparse.empty((n_hidden_neurons,))
    )


def generate_readout_state(n_output_neurons: int):
    return ReadoutState(
        accumulated_spikes=jnp.zeros(n_output_neurons, dtype=jnp.float32)
    )


def step(params: LIFParams, state: LIFState, inputs: sparse.BCOO):
    incoming_current = inputs @ params.W_in + state.spike @ params.W_rec
    decayed_v = params.alpha * state.v
    reset_term = state.spike.todense() * params.v_thresh
    v = decayed_v + incoming_current.todense() - reset_term
    spike = heaviside(v - params.v_thresh)
    spike = sparse.BCOO.fromdense(spike)
    # NOTE we subtract on v[t] for s[t-1]. other implementations subtract for v[t+1] on s[t]
    # does it matter? no idea
    # NOTE also, if we don't reset to 0, a huge weight will make the neuron spike
    # more than once. this is fine in principle, but does the gradient account for
    # that?

    return LIFState(v=v, spike=spike)


def readout_step(params: ReadoutParams, state: ReadoutState, spike: sparse.BCOO):
    weighted_inputs = spike @ params.W_in
    accumulated_spikes = state.accumulated_spikes + weighted_inputs.todense()
    return ReadoutState(accumulated_spikes=accumulated_spikes)
