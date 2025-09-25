from jax import random
import jax.numpy as jnp
from jax.experimental import sparse
from sparse_lif import (
    LIFParams,
    ReadoutParams,
    generate_lif_params,
    generate_readout_params,
    generate_lif_state,
    generate_readout_state,
    readout_step,
    step,
)


def test_lif_params_dimensions():
    key = random.PRNGKey(0)
    n_input_neurons = 10
    n_hidden_neurons = 20
    initial_decay = 0.9
    params = generate_lif_params(
        key, n_input_neurons, n_hidden_neurons, initial_decay=initial_decay
    )

    assert isinstance(params, LIFParams)
    assert params.alpha.shape == (n_hidden_neurons,)
    assert params.v_thresh.shape == (n_hidden_neurons,)
    assert params.W_in.shape == (n_input_neurons, n_hidden_neurons)
    assert params.W_rec.shape == (n_hidden_neurons, n_hidden_neurons)

    assert jnp.all(params.alpha == initial_decay)
    assert jnp.all(params.v_thresh == 1.0)


def test_lif_params_with_recurrent():
    key = random.PRNGKey(0)
    n_input_neurons = 10
    n_hidden_neurons = 20

    params = generate_lif_params(
        key, n_input_neurons, n_hidden_neurons, has_recurrent_connections=True
    )

    assert params.W_rec.shape == (n_hidden_neurons, n_hidden_neurons)


def test_readout_params_dimensions():
    key = random.PRNGKey(0)
    n_input_neurons = 20
    n_output_neurons = 5

    params = generate_readout_params(key, n_input_neurons, n_output_neurons)

    assert isinstance(params, ReadoutParams)
    assert params.W_in.shape == (n_input_neurons, n_output_neurons)


def test_lif_state_dimensions():
    key = random.PRNGKey(0)
    n_hidden_neurons = 20

    state = generate_lif_state(key, n_hidden_neurons)

    assert state.v.shape == (n_hidden_neurons,)
    assert state.spike.shape == (n_hidden_neurons,)
    assert jnp.all(state.v == 0.0)


def test_readout_state_dimensions():
    n_output_neurons = 5

    state = generate_readout_state(n_output_neurons)

    assert state.accumulated_spikes.shape == (n_output_neurons,)
    assert jnp.all(state.accumulated_spikes == 0.0)


def test_readout_step_only_affects_connected_neurons():
    n_input_neurons = 5
    n_output_neurons = 2
    weights = jnp.array([1.0])
    indeces = jnp.array([[4, 1]])
    W_in = sparse.BCOO((weights, indeces), shape=(n_input_neurons, n_output_neurons))
    readout_params = ReadoutParams(W_in=W_in)

    readout_state = generate_readout_state(n_output_neurons)
    spike = sparse.BCOO.fromdense(jnp.ones(n_input_neurons))
    readout_state = readout_step(readout_params, readout_state, spike)

    assert readout_state.accumulated_spikes[0] == 0.0
    assert readout_state.accumulated_spikes[1] == 1.0


def test_lif_step_only_affects_connected_neurons():
    n_input_neurons = 5
    n_hidden_neurons = 3
    weights_in = jnp.array([1.2])
    # input connection from input 4 to hidden 1
    indeces_in = jnp.array([[4, 1]])
    W_in = sparse.BCOO(
        (weights_in, indeces_in), shape=(n_input_neurons, n_hidden_neurons)
    )
    weights_rec = jnp.array([1.2])
    # reccurent connection from hidden 1 to hidden 3
    indeces_rec = jnp.array([[1, 2]])
    W_rec = sparse.BCOO(
        (weights_rec, indeces_rec), shape=(n_hidden_neurons, n_hidden_neurons)
    )
    alpha = jnp.ones((n_hidden_neurons,)) * 0.9
    v_thresh = jnp.ones((n_hidden_neurons,)) * 1.0
    lif_params = LIFParams(W_in=W_in, W_rec=W_rec, alpha=alpha, v_thresh=v_thresh)

    lif_state = generate_lif_state(random.PRNGKey(0), n_hidden_neurons)
    full_input_spikes = sparse.BCOO.fromdense(jnp.ones(n_input_neurons))
    lif_state = step(lif_params, lif_state, full_input_spikes)

    # we have an input connection from input 4 to hidden 1
    # so we expect hidden 1 to spike but none of the others
    dense_spike = lif_state.spike.todense()
    assert dense_spike[1] == 1.0
    assert jnp.sum(dense_spike) == 1.0

    no_input_spikes = sparse.empty((n_input_neurons,))
    lif_state = step(lif_params, lif_state, no_input_spikes)
    dense_spike = lif_state.spike.todense()
    # at the next step, we dont get inputs
    # only the current connection will have an effect now
    assert dense_spike[2] == 1.0
    assert jnp.sum(dense_spike) == 1.0
