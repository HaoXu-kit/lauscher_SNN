from __future__ import annotations
import os, sys, subprocess, tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import jax.numpy as jnp
from jax import random
from datasets import load_dataset, Audio
import soundfile as sf
import wandb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
 
# ========= 配置区域：按需改动 =========
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
JAX_SNN_DIR = Path(__file__).parent / "jax_snn_guide"  # 指向你的 jax_snn 仓库根目录
T = 200              # 时间步数（和你的 SNN 时间展开一致）
C = 70               # 输入通道数（launcher 的 --num_channels）
EPOCHS = 50
LR = 1e-3
HIDDEN = [256]       # 隐藏层神经元数（可多层，如 [256,128]）
RECURRENT = [True]   # 与 HIDDEN 等长；每层是否有循环连接
GSC_VERSION = "v0.02"
SAMPLE_RATE = 16000
TRAIN_PLUS_VAL = True  # 训练集= train+validation
VERBOSE = False
USE_CACHE = False      # 设 True 可把构建好的数据缓存到本地 npz
DEBUG_SMALL_DATA = True     # True 时只取少量样本，方便本地 CPU 调试
DISABLE_WANDB_IN_DEBUG = True   # 当 debug 模式下禁用 wandb logging
SMALL_TRAIN_N = 40   # 调试时使用的训练样本数
SMALL_TEST_N  = 20   # 调试时使用的测试样本数
CACHE_PATH = Path(__file__).parent / f"gsc_T{T}_C{C}.npz"
PARALLEL_WORKERS = min(os.cpu_count() or 8, 128)  # 并行进程数
# ====================================
 
# 让脚本能 import 到 jax_snn（不改仓库本身）
if str(JAX_SNN_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_SNN_DIR))
 
# Sanity check: ensure expected files exist under JAX_SNN_DIR
assert (JAX_SNN_DIR / "training.py").exists(), f"training.py not found under {JAX_SNN_DIR}. Fix JAX_SNN_DIR or your layout."
 
try:
    # Prefer unqualified imports so training.py and this script refer to the SAME modules
    from training import run_training_loop, evaluate_network  # type: ignore
    from models.lif_network import (  # type: ignore
        generate_lif_network_params,
        generate_lif_network_state,
    )
except ImportError:
    # Fallback: allow package-qualified imports if needed
    ROOT = JAX_SNN_DIR.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from jax_snn_guide.training import run_training_loop, evaluate_network  # type: ignore
    from jax_snn_guide.models.lif_network import (  # type: ignore
        generate_lif_network_params,
        generate_lif_network_state,
    )
 
# ----------------- launcher 调用 & 稀疏→稠密（numpy 实现） -----------------
 
def _run_launcher(audio_path: str | Path, out_npz: str | Path, num_channels: int, verbose: bool):
    cmd = [sys.executable, "-m", "lauscher", str(audio_path), str(out_npz),
           "--num_channels", str(num_channels)]
    if verbose:
        cmd.append("--verbose")
    subprocess.run(cmd, check=True)
 
def _load_sparse_events(npz_path: str | Path) -> np.ndarray:
    """返回 (2,K)：row0=times(sec, float64), row1=labels(int)"""
    d = np.load(npz_path, allow_pickle=False)
    key = "spikes" if "spikes" in d.files else d.files[0]
    return np.asarray(d[key], dtype=np.float64)
 
def _sparse_to_dense_numpy(events_2xK: np.ndarray, T: int, C: int, max_time_sec: float) -> np.ndarray:
    """把 (2,K) 稀疏事件转成 [T,C] 稠密 0/1（float32）"""
    dense = np.zeros((T, C), dtype=np.float32)
    if events_2xK.size == 0:
        return dense
    times = events_2xK[0]                                   # 秒
    labels = np.clip(events_2xK[1].astype(np.int64), 0, C - 1)
    dt = (max_time_sec / float(T)) + 1e-12
    bins = np.floor(times / dt).astype(np.int64)
    bins = np.clip(bins, 0, T - 1)
    dense[bins, labels] = 1.0
    return dense
 
def encode_waveform_to_dense_numpy(wav: np.ndarray, sr: int, T: int, C: int, verbose: bool=False) -> np.ndarray:
    """waveform(np.float32) → launcher 稀疏事件 → 稠密 [T,C]（numpy float32）"""
    max_time_sec = float(len(wav)) / float(sr)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f_wav, \
         tempfile.NamedTemporaryFile(suffix=".npz") as f_npz:
        sf.write(f_wav.name, wav, sr)
        _run_launcher(f_wav.name, f_npz.name, num_channels=C, verbose=verbose)
        events = _load_sparse_events(f_npz.name)
    return _sparse_to_dense_numpy(events, T=T, C=C, max_time_sec=max_time_sec)
 
def _worker_encode_example(ex: dict, T: int, C: int, verbose: bool, label2id: dict) -> tuple[np.ndarray, np.int32]:
    """
    进程池工作函数：对单个样本做编码并返回 (x, y)
    ex: datasets 的一个样本字典，包含 ex["audio"]["array"], ex["audio"]["sampling_rate"], ex["label"]
    """
    wav = ex["audio"]["array"]  # np.float32 [samples]
    sr = ex["audio"]["sampling_rate"]
    x = encode_waveform_to_dense_numpy(wav, sr, T=T, C=C, verbose=verbose)  # [T,C] float32
    y = np.int32(label2id[ex["label"]])
    return x, y
 
# ----------------- 构建数据（不依赖 jax_snn 内部） -----------------
 
def build_gsc_numpy(T: int, C: int,
                    version: str = GSC_VERSION,
                    sample_rate: int = SAMPLE_RATE,
                    train_plus_val: bool = TRAIN_PLUS_VAL,
                    verbose: bool = VERBOSE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if USE_CACHE and CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"], int(z["n_classes"])
 
    ds = load_dataset("google/speech_commands", version, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
 
    from datasets import concatenate_datasets
 
    if train_plus_val:
        train_data = concatenate_datasets([ds["train"], ds["validation"]])
    else:
        train_data = ds["train"]
 
    test_data = ds["test"]
 
    if DEBUG_SMALL_DATA:
        train_data = train_data.select(range(min(SMALL_TRAIN_N, len(train_data))))
        test_data  = test_data.select(range(min(SMALL_TEST_N, len(test_data))))
 
    # Build label mapping using labels from all splits to avoid KeyError on test-only labels
    labels_train = set(ds["train"]["label"]) if "train" in ds else set()
    labels_val   = set(ds["validation"]["label"]) if "validation" in ds else set()
    labels_test  = set(ds["test"]["label"]) if "test" in ds else set()
    all_labels = sorted(labels_train | labels_val | labels_test)
    label2id = {l: i for i, l in enumerate(all_labels)}
 
    Xtr_list, ytr_list = [], []
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = [pool.submit(_worker_encode_example, ex, T, C, verbose, label2id) for ex in train_data]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Encoding train (workers={PARALLEL_WORKERS})"):
            x, y = fut.result()
            Xtr_list.append(x)
            ytr_list.append(y)
 
    Xte_list, yte_list = [], []
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = [pool.submit(_worker_encode_example, ex, T, C, verbose, label2id) for ex in test_data]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Encoding test (workers={PARALLEL_WORKERS})"):
            x, y = fut.result()
            Xte_list.append(x)
            yte_list.append(y)
 
    Xtr = np.stack(Xtr_list, 0).astype(np.float32)
    ytr = np.array(ytr_list, dtype=np.int32)
    Xte = np.stack(Xte_list, 0).astype(np.float32)
    yte = np.array(yte_list, dtype=np.int32)
    n_classes = len(all_labels)
 
    if USE_CACHE:
        np.savez_compressed(CACHE_PATH, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_classes=n_classes)
 
    return Xtr, ytr, Xte, yte, n_classes
 
# ----------------- 主流程：零改动调用 jax_snn 训练 -----------------
 
def main():
    Xtr, ytr, Xte, yte, n_classes = build_gsc_numpy(T=T, C=C)
    data = (jnp.asarray(Xtr), jnp.asarray(ytr), jnp.asarray(Xte), jnp.asarray(yte))
 
    key = random.PRNGKey(0)
    params = generate_lif_network_params(
        key,
        n_inputs=C,                        # 输入维度=C（launcher 通道数）
        hidden_neuron_counts=HIDDEN,
        hidden_recurrent_config=RECURRENT,
        n_outputs=n_classes,
    )
    state = generate_lif_network_state(key, params)
 
    if DEBUG_SMALL_DATA and DISABLE_WANDB_IN_DEBUG:
        os.environ["WANDB_MODE"] = "disabled"
        print("[INFO] DEBUG 模式：wandb 以 disabled 模式初始化，避免 wandb.log 报错")
        wandb.init(project="snn-gsc",
                   mode="disabled",
                   config=dict(T=T, C=C, hidden=HIDDEN, recurrent=RECURRENT, lr=LR))
    else:
        wandb.init(project="snn-gsc",
                   config=dict(T=T, C=C, hidden=HIDDEN, recurrent=RECURRENT, lr=LR))
    _ = run_training_loop(
        initial_network_params=params,
        initial_network_state=state,
        n_epochs=EPOCHS,
        initial_epoch=0,
        num_training_samples=(40 if DEBUG_SMALL_DATA else len(ytr)),
        data=data,
        learning_rate=LR,
        eval_fn=evaluate_network,
        key=key,
        learning_rule=None,  # None -> 默认 BPTT
    )
 
if __name__ == "__main__":
    main()