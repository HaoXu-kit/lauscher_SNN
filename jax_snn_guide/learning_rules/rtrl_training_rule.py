from functools import partial
from jax import jit, tree, grad, lax
from jax import numpy as jnp

from learning_rules.losses import cross_entropy_loss

# TODO generalize to any kind of network

from models.lif_network import LIFNetworkParams, LIFNetworkState, lif_network_step
from learning_rules.learning_rule import LearningRule, update_rule


class RTRLlearningRule(LearningRule):
    def grad_func(self, network_params, network_state, x, y_true):
        return grad(calculate_loss_for_sample, argnums=0)(
            network_params, network_state, x, y_true
        )

    def update_params(self, network_params, grads, learning_rate):
        bound_update_rule = partial(update_rule, learning_rate)
        return tree.map_with_path(bound_update_rule, network_params, grads)


@jit
def calculate_loss_for_sample(
    network_params: LIFNetworkParams,
    initial_network_state: LIFNetworkState,
    inputs: jnp.ndarray,
    y_true: int,
):
    bound_network_step = partial(lif_network_step, network_params)
    final_state, _ = lax.scan(bound_network_step, initial_network_state, inputs)
    logits = final_state.output_state.accumulated_spikes

    current_loss = cross_entropy_loss(logits, y_true)
    return current_loss
