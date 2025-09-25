from jax import random
import jax.numpy as jnp
from models.lif import (
    generate_lif_params,
    generate_lif_state,
    lif_step,
    LIFParams,
    LIFState,
)


def test_lif_params_dimensions():
    key = random.PRNGKey(0)
    n_input_neurons = 10
    n_hidden_neurons = 20
    initial_decay = 0.96
    params = generate_lif_params(key, n_input_neurons, n_hidden_neurons)

    assert isinstance(params, LIFParams)
    assert params.alpha.shape == (n_hidden_neurons,)
    assert params.v_thresh.shape == (n_hidden_neurons,)
    assert params.W_in.shape == (n_input_neurons, n_hidden_neurons)
    assert params.W_in_mask.shape == (n_input_neurons, n_hidden_neurons)
    assert params.W_rec.shape == (n_hidden_neurons, n_hidden_neurons)
    assert params.W_rec_mask.shape == (n_hidden_neurons, n_hidden_neurons)
    assert jnp.all(params.alpha == initial_decay)
    assert jnp.all(params.v_thresh == 1.0)


def test_lif_state_dimensions():
    key = random.PRNGKey(0)
    n_hidden_neurons = 20
    state = generate_lif_state(key, n_hidden_neurons)

    assert isinstance(state, LIFState)
    assert state.v.shape == (1, n_hidden_neurons)
    assert state.spike.shape == (1, n_hidden_neurons)
    assert jnp.all(state.v == 0.0)


def test_step_function():
    key = random.PRNGKey(0)
    n_input_neurons = 10
    n_hidden_neurons = 20
    params = generate_lif_params(key, n_input_neurons, n_hidden_neurons)
    state = generate_lif_state(key, n_hidden_neurons)
    inputs = jnp.ones((1, n_input_neurons))

    assert isinstance(state, LIFState)
    assert state.v.shape == (1, n_hidden_neurons)
    assert state.spike.shape == (1, n_hidden_neurons)
    assert jnp.all(state.v == 0.0)

    new_state = lif_step(params, state, inputs)

    assert isinstance(new_state, LIFState)
    assert new_state.v.shape == (1, n_hidden_neurons)
    assert new_state.spike.shape == (1, n_hidden_neurons)


def test_step_function_for_single_neuron():
    key = random.PRNGKey(0)
    n_input_neurons = 1
    n_hidden_neurons = 1
    params = generate_lif_params(key, n_input_neurons, n_hidden_neurons)
    params = params._replace(W_in=jnp.array([[0.1]]))
    state = generate_lif_state(key, n_hidden_neurons)
    inputs = jnp.ones((1, n_input_neurons))
    assert jnp.all(state.v == 0)
    new_state = lif_step(params, state, inputs)
    assert jnp.all(new_state.v >= 0)
    assert jnp.all(new_state.v <= 0.1)
