import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
from lauscher.spike_train import SpikeTrain  

def _load_npz_as_spiketrain(p: str | Path) -> SpikeTrain:
    d = np.load(p, allow_pickle=False)
    key = 'spikes' if 'spikes' in d.files else (d.files[0])
    return SpikeTrain.from_sparse(d[key])

def encode_via_launcher_to_tensor( 
    audio_path: str | Path,
    *,
    num_channels: int = 70,
    device: str = "cpu",
    verbose: bool = False,
):
    """调用官方 launcher（临时 .npz），立即返回一个稀疏事件张量，形状为 (2, K)，
    其中第0行是脉冲时间（秒，float），第1行是脉冲标签（通道ID，int）。"""
    with tempfile.TemporaryDirectory() as tdir:
        out = Path(tdir) / "tmp_spikes.npz"
        cmd = [sys.executable, "-m", "lauscher", str(audio_path), str(out),
               "--num_channels", str(num_channels)]
        if verbose:
            cmd.append("--verbose")
        subprocess.run(cmd, check=True)
        st = _load_npz_as_spiketrain(out)
        return st.to_torch(device=device)  # -> events_2xK: torch.Tensor with shape (2, K)
    
def encode_folder_via_launcher_to_tensor(
    folder: str | Path,
    *,
    num_channels: int = 70,
    device: str = "cpu",
    verbose: bool = False,
    patterns: tuple[str, ...] = (".flac", ".wav"),
    recursive: bool = True,
    return_paths: bool = False,
):
    """批量把文件夹中的音频编码为 PyTorch 张量。

    Args:
        folder: 目录路径。
        num_channels: 传入 launcher 的通道数。
        device: 目标设备（"cpu"/"cuda"/...）。
        verbose: 是否打印详细信息。
        patterns: 需要匹配的文件后缀元组（包含点），默认同时支持 .flac / .wav。
        recursive: 是否递归扫描子目录。
        return_paths: 若为 True，则返回 (path, tensor)；否则返回 tensor。

    Returns:
        list[torch.Tensor] 或 list[tuple]: 返回稀疏事件张量列表，形状均为 (2, K)，
        或者当 return_paths=True 时返回 (path, tensor) 元组列表。
    """
    folder = Path(folder)
    if not folder.exists():
        if verbose:
            print(f"[WARN] Folder not found: {folder}")
        return []

    # 收集文件（支持多后缀 & 可递归）
    files: list[Path] = []
    for ext in patterns:
        pattern = f"*{ext}"
        files.extend(folder.rglob(pattern) if recursive else folder.glob(pattern))
    # 去重并排序，且只保留文件
    seen = set()
    unique_files: list[Path] = []
    for p in sorted(files):
        if p.is_file() and p not in seen:
            seen.add(p)
            unique_files.append(p)

    if not unique_files:
        if verbose:
            print(f"[WARN] No audio files matched in {folder} with patterns {patterns}")
        return []

    results = []
    for f in unique_files:
        if verbose:
            print(f"[INFO] Encoding {f}")
        events_2xK = encode_via_launcher_to_tensor(
            audio_path=str(f),
            num_channels=num_channels,
            device=device,
            verbose=verbose,
        )
        results.append((f, events_2xK) if return_paths else events_2xK)

    return results