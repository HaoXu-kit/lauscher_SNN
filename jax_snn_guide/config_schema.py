from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 1e-3


class NetworkConfig(BaseModel):
    layer_sizes: List[int] = Field(default_factory=lambda: [1024, 1024])


class DatasetConfig(BaseModel):
    name: Literal["shd", "n-mnist"] = "shd"
    batch_size: int = 1


class ExperimentConfig(BaseModel):
    project_name: str = "snn_comparison"
    learning_rule: Literal["bptt", "e-prop", "decolle"] = "bptt"
    total_epochs: int = 100
    seed: int = 42
    wandb_run_id: Optional[str] = None

    optimizer: OptimizerConfig = OptimizerConfig()
    network: NetworkConfig = NetworkConfig()
    dataset: DatasetConfig = DatasetConfig()
