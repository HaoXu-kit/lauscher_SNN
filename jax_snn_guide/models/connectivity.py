import jax.numpy as jnp
from jax import random

CONNECTIVITY = 1.0

def generate_input_weights(key, n_input_neurons: int, n_hidden_neurons: int):
    key_in, key_in_mask = random.split(key, 2)
     # Xavier/Glorot uniform initialization for W_in
    limit_in = jnp.sqrt(6.0 / (n_input_neurons + n_hidden_neurons))
    W_in = random.uniform(
        key_in, (n_input_neurons, n_hidden_neurons), minval=-limit_in, maxval=limit_in
    )
    W_in_mask = random.bernoulli(
        key_in_mask, CONNECTIVITY, (n_input_neurons, n_hidden_neurons)
    ).astype(jnp.float32)
    return W_in, W_in_mask

def generate_recurrent_weights(key, n_hidden_neurons: int):
    key_rec, key_rec_mask = random.split(key, 2)
    # Xavier/Glorot uniform initialization for W_rec (if used)
    limit_rec = jnp.sqrt(6.0 / (n_hidden_neurons + n_hidden_neurons))
    W_rec = random.uniform(
        key_rec,
        (n_hidden_neurons, n_hidden_neurons),
        minval=-limit_rec,
        maxval=limit_rec,
    )
    W_rec = W_rec.at[jnp.diag_indices_from(W_rec)].set(0.0)
    W_rec_mask = random.bernoulli(
        key_rec_mask, CONNECTIVITY, (n_hidden_neurons, n_hidden_neurons)
    ).astype(jnp.float32)
    W_rec_mask = W_rec_mask.at[jnp.diag_indices_from(W_rec_mask)].set(0.0)
    return W_rec, W_rec_mask