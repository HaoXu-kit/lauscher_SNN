from jax import jit, nn
from jax import numpy as jnp


@jit
def cross_entropy_loss(logits: jnp.ndarray, y_true: int) -> jnp.ndarray:
    log_probs = nn.log_softmax(logits)
    y_true_one_hot = nn.one_hot(y_true, num_classes=logits.shape[-1])
    return -jnp.sum(y_true_one_hot * log_probs)
