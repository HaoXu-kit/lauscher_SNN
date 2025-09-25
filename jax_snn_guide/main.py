import os
import pickle

from models.lif_network import (
    generate_lif_network_params,
    generate_lif_network_state,
)
from training import (
    evaluate_network,
    run_training_loop,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import hydra
import wandb
from omegaconf import OmegaConf

from config_schema import ExperimentConfig

from jax import random


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: ExperimentConfig):
    USE_WANDB = False
    n_steps = 100

    if cfg.dataset.name == "shd":
        n_output_neurons = 20
        n_features = 700

        from shd_data_loader import load_preprocessed

        X_train, y_train, X_test, y_test = load_preprocessed(n_steps)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    if "wandb_run_id" in cfg and cfg.wandb_run_id:
        run_id = cfg.wandb_run_id
        resume_status = "must"
    else:
        run_id = wandb.util.generate_id()
        resume_status = None

    wandb.init(
        project="jax-snn",
        entity="xh63491181-karlsruhe-institute-of-technology",
        name=f"{run_id}",
        id=f"{run_id}",
        resume=resume_status,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if not USE_WANDB else "online",
    )

    key = random.PRNGKey(2)
    key, params_key, state_key, training_key = random.split(key, 4)

    initial_epoch = 0
    if wandb.run.resumed:
        print("Resuming run, restoring model weights...")
        artifact = wandb.use_artifact("model-weights:latest")
        artifact_dir = artifact.download()
        weights_path = os.path.join(artifact_dir, "network_weights.pkl")
        with open(weights_path, "rb") as f:
            network_params = pickle.load(f)
        initial_epoch = wandb.run.step
    else:
        network_params = generate_lif_network_params(
            params_key,
            n_features,
            cfg.network.layer_sizes,
            cfg.network.layer_recurrent_flags,
            n_output_neurons,
        )

    network_state = generate_lif_network_state(state_key, network_params)

    print("Starting training...")
    try:
        learning_rate = cfg.optimizer.learning_rate

        network_params = run_training_loop(
            initial_network_params=network_params,
            initial_network_state=network_state,
            n_epochs=cfg.total_epochs,
            initial_epoch=initial_epoch,
            num_training_samples=100,
            data=(X_train, y_train, X_test, y_test),
            learning_rate=learning_rate,
            eval_fn=evaluate_network,
            key=training_key,
        )

    finally:
        if USE_WANDB:
            weights_path = f"{wandb.run.dir}/network_weights.pkl"
            artifact_name = f"{wandb.run.name}_weights"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(weights_path)
            wandb.log_artifact(artifact)
            wandb.finish()


if __name__ == "__main__":
    main()
