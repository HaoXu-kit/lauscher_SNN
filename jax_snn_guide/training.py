from functools import partial
import pickle
from jax import jit, lax, vmap, random
from jax import numpy as jnp
import wandb

from learning_rules.losses import cross_entropy_loss
from models.lif_network import LIFNetworkParams, LIFNetworkState, lif_network_step
from learning_rules.learning_rule import LearningRule


@jit
def evaluate(
    network_params: LIFNetworkParams,
    network_state: LIFNetworkState,
    input_sequences: jnp.ndarray,
    labels: jnp.ndarray,
):
    """
    Evaluates the network on a batch of input sequences and labels.
    Returns mean loss and mean accuracy.
    Handles both single-sample and batch evaluation.
    """

    def eval_step(inputs, label):
        bound_network_step = partial(lif_network_step, network_params)
        final_state, _ = lax.scan(bound_network_step, network_state, inputs)
        logits = final_state.output_state.accumulated_spikes
        loss = cross_entropy_loss(logits, label)
        pred = jnp.argmax(logits)
        acc = jnp.where(pred == label, 1.0, 0.0)
        return loss, acc

    # If input_sequences is 2D (single sample), add batch dimension
    if input_sequences.ndim == 2:
        input_sequences = input_sequences[None, ...]
        labels = jnp.array([labels])

    losses, accuracies = vmap(eval_step)(input_sequences, labels)
    return jnp.mean(losses), jnp.mean(accuracies)


def evaluate_network(
    network_params: LIFNetworkParams,
    network_state: LIFNetworkState,
    key: jnp.ndarray,
    eval_X_dense: jnp.ndarray,
    eval_y_labels: jnp.ndarray,
    num_eval_samples: int = 30,
):
    """
    Evaluates the network on a number of samples.
    """
    indices = random.choice(key, len(eval_y_labels), (num_eval_samples,), replace=False)

    batch_input_sequences = eval_X_dense[indices]
    batch_input_sequences = batch_input_sequences.astype(jnp.float32)
    batch_labels = eval_y_labels[indices]

    mean_loss_eval, mean_accuracy = evaluate(
        network_params, network_state, batch_input_sequences, batch_labels
    )
    return mean_loss_eval, mean_accuracy


def train_on_sample(
    x: jnp.ndarray,
    y_true: int,
    learning_rule: LearningRule,
    network_params: LIFNetworkParams,
    network_state: LIFNetworkState,
    learning_rate: float,
):
    grads = learning_rule.grad_func(network_params, network_state, x, y_true)
    updated_network_params = learning_rule.update_params(
        network_params, grads, learning_rate
    )
    return updated_network_params


def epoch_update(
    network_params,
    learning_rule,
    X_train,
    y_train,
    shuffled_indices,
    network_state,
    learning_rate,
    num_training_samples,
):
    def fori_train_step(i, params):
        x = X_train[shuffled_indices[i]]
        y = y_train[shuffled_indices[i]]
        return train_on_sample(
            x, y, learning_rule, params, network_state, learning_rate
        )

    return lax.fori_loop(0, num_training_samples, fori_train_step, network_params)


def run_training_loop(
    initial_network_params: LIFNetworkParams,
    initial_network_state: LIFNetworkState,
    n_epochs: int,
    initial_epoch: int,
    num_training_samples: int,
    data: tuple,
    learning_rate: float,
    eval_fn,
    key: jnp.ndarray,
    learning_rule: LearningRule = None,
):
    X_train, y_train, X_test, y_test = data
    X_train = X_train.astype(jnp.float32)
    X_test = X_test.astype(jnp.float32)
    network_params = initial_network_params
    network_state = initial_network_state
    n_eval_samples = 30

    num_total_samples = len(y_train)
    if learning_rule is None:
        from learning_rules.bptt_learning_rule import BPTTLearningRule

        learning_rule = BPTTLearningRule()
    jitted_epoch_update = jit(epoch_update, static_argnames=["learning_rule"])

    for epoch in range(initial_epoch, n_epochs):
        key, shuffle_key, eval_key = random.split(key, 3)
        shuffled_indices = random.permutation(shuffle_key, num_total_samples)

        network_params = jitted_epoch_update(
            network_params,
            learning_rule,
            X_train,
            y_train,
            shuffled_indices,
            network_state,
            learning_rate,
            num_training_samples,
        )

        # Evaluate on a subset of the training data
        current_train_loss, train_accuracy = eval_fn(
            network_params,
            network_state,
            eval_key,
            X_train,
            y_train,
            n_eval_samples,
        )

        # Evaluate on the full test set
        num_test_samples = len(y_test)
        current_test_loss, test_accuracy = eval_fn(
            network_params,
            network_state,
            eval_key,
            X_test,
            y_test,
            num_test_samples,
        )

        print(
            f"After epoch {epoch}: "
            f"Train Loss = {current_train_loss:.4f}, Train Acc = {train_accuracy:.4f} | "
            f"Test Loss = {current_test_loss:.4f}, Test Acc = {test_accuracy:.4f}"
        )
        wandb.log(
            {
                "train_loss": current_train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": current_test_loss,
                "test_accuracy": test_accuracy,
            },
            step=epoch,
        )

        if epoch % 100 == 0:
            weights_path = f"{wandb.run.dir}/network_weights.pkl"
            with open(weights_path, "wb") as f:
                pickle.dump(network_params, f)

    return network_params
