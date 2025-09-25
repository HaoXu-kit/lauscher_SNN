from typing import NamedTuple
from jax import random
from jax import numpy as jnp
from models.lif import (
    LIFParams,
    LIFState,
    generate_lif_params,
    generate_lif_state,
    lif_step,
)
from models.readout import (
    ReadoutParams,
    ReadoutState,
    generate_readout_params,
    generate_readout_state,
    readout_step,
)


class LIFNetworkParams(NamedTuple):
    hidden_params: list[LIFParams]
    output_params: ReadoutParams


class LIFNetworkState(NamedTuple):
    hidden_states: list[LIFState]
    output_state: ReadoutState


def generate_lif_network_params(
    key: random.PRNGKey,
    n_inputs,
    hidden_neuron_counts,
    hidden_recurrent_config,
    n_outputs,
) -> LIFNetworkParams:
    key, output_param_key = random.split(key, 2)

    hidden_params = []
    current_input_dim = n_inputs
    for i, num_neurons_in_layer in enumerate(hidden_neuron_counts):
        key, layer_param_key = random.split(key, 2)
        params = generate_lif_params(
            layer_param_key,
            current_input_dim,
            num_neurons_in_layer,
            has_recurrent_connections=hidden_recurrent_config[i],
        )
        hidden_params.append(params)

        current_input_dim = num_neurons_in_layer

    output_params = generate_readout_params(
        output_param_key,
        current_input_dim,
        n_outputs,
    )
    return LIFNetworkParams(hidden_params, output_params)


def generate_lif_network_state(
    key, network_params: LIFNetworkParams
) -> LIFNetworkState:
    layer_keys = random.split(key, len(network_params.hidden_params))
    hidden_states = [
        generate_lif_state(layer_keys[idx], params.alpha.shape[-1])
        for idx, params in enumerate(network_params.hidden_params)
    ]
    output_state = generate_readout_state(network_params.output_params.W_in.shape[-1])
    network_state = LIFNetworkState(hidden_states, output_state)
    return network_state


def lif_network_step(
    network_params: LIFNetworkParams,
    carry_state: LIFNetworkState,
    input_slice: jnp.ndarray,
):
    new_hidden_states = []
    current_layer_input_spikes = input_slice

    num_hidden_layers = len(network_params.hidden_params)
    if num_hidden_layers > 0:
        # NOTE yes it's a loop in jax, but we can't use lax.scan because layers can have different sizes
        for i in range(num_hidden_layers):
            hidden_state = lif_step(
                network_params.hidden_params[i],
                carry_state.hidden_states[i],
                current_layer_input_spikes,
            )
            new_hidden_states.append(hidden_state)
            current_layer_input_spikes = hidden_state.spike

    new_output_state = readout_step(
        network_params.output_params,
        carry_state.output_state,
        current_layer_input_spikes,
    )
    new_state = LIFNetworkState(new_hidden_states, new_output_state)

    return new_state, None
