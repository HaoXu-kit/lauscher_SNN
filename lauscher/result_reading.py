#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple

def infer_spikes(npz: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 NPZ 中抽取 spike_times (sec) 和 spike_channels (int).
    优先级：
      1) 显式 keys: (times + channels)
      2) per-channel: times_list / spiketrains / spikes（object数组）
      3) 单数组 arr_0 等：若形状为 (2, N) 或 (N, 2)，按 [time, ch] 解析
      4) 只有 times：通道全置 0
    """
    keys = set(npz.files)

    # 1) 扁平 keys
    tk = next((k for k in ['times','time','t','spike_times'] if k in keys), None)
    ck = next((k for k in ['channels','channel','ch','neurons','units','unit'] if k in keys), None)
    if tk is not None and ck is not None:
        times = np.asarray(npz[tk], dtype=float).ravel()
        chans = np.asarray(npz[ck], dtype=int).ravel()
        if times.shape == chans.shape:
            return times, chans

    # 2) 每通道列表
    lk = next((k for k in ['times_list','spikes','spiketrains','spike_trains'] if k in keys), None)
    if lk is not None:
        arr = npz[lk]
        times_all, chans_all = [], []
        for ch, t in enumerate(arr):
            if t is None: 
                continue
            t = np.asarray(t, dtype=float).ravel()
            if t.size == 0:
                continue
            times_all.append(t)
            chans_all.append(np.full_like(t, ch, dtype=int))
        if times_all:
            return np.concatenate(times_all), np.concatenate(chans_all)

    # 3) 兼容你现在的 arr_0: (2, N) 或 (N, 2)
    ak = next(iter(keys))  # 只有一个键时直接取
    arr = np.asarray(npz[ak])
    if arr.ndim == 2:
        if arr.shape[0] == 2:           # (2, N)
            times = arr[0, :].astype(float).ravel()
            chans = arr[1, :].astype(int).ravel()
            return times, chans
        if arr.shape[1] == 2:           # (N, 2)
            times = arr[:, 0].astype(float).ravel()
            chans = arr[:, 1].astype(int).ravel()
            return times, chans

    # 4) 只有 times
    if tk is not None and ck is None:
        t = np.asarray(npz[tk], dtype=float).ravel()
        c = np.zeros_like(t, dtype=int)
        return t, c

    raise ValueError(f"无法从 {list(keys)} 中解析出 (times, channels)。")

def main():
    p = argparse.ArgumentParser(description="Visualize spike trains in NPZ (raster + PSTH).")
    p.add_argument("npz_path", help="输入的 .npz 文件路径")
    p.add_argument("--bin", type=float, default=0.01, help="PSTH 时间窗(秒)，默认0.01")
    p.add_argument("--out", type=str, default=None, help="保存图片文件名前缀（如 output.png）")
    p.add_argument("--title", type=str, default=None, help="图标题")
    args = p.parse_args()

    if not os.path.isfile(args.npz_path):
        raise FileNotFoundError(args.npz_path)

    data = np.load(args.npz_path, allow_pickle=True)
    print("NPZ keys:", data.files)

    times, chans = infer_spikes(data)

    # 排序便于显示
    idx = np.argsort(times)
    times, chans = times[idx], chans[idx]

    tmin = float(np.min(times)) if times.size else 0.0
    tmax = float(np.max(times)) if times.size else 0.0
    n_spikes = int(times.size)
    n_channels = int(np.max(chans) + 1) if chans.size else 1
    print(f"Spikes={n_spikes}, Channels≈{n_channels}, Time=[{tmin:.6f},{tmax:.6f}]s")

    # 1) Raster
    plt.figure(figsize=(10, 4))
    plt.plot(times, chans, linestyle="None", marker=".", markersize=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(args.title or "Spike Raster")
    plt.tight_layout()
    if args.out:
        base, ext = os.path.splitext(args.out)
        out_raster = f"{base}_raster{ext or '.png'}"
        plt.savefig(out_raster, dpi=150)
        print("Saved:", out_raster)
    else:
        plt.show()

    # 2) PSTH
    if tmax > tmin and n_spikes > 0:
        bins = np.arange(tmin, tmax + args.bin, args.bin)
        counts, edges = np.histogram(times, bins=bins)
        plt.figure(figsize=(10, 3))
        plt.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Spike count / {args.bin:g}s")
        plt.title(args.title or "PSTH")
        plt.tight_layout()
        if args.out:
            out_psth = f"{base}_psth{ext or '.png'}"
            plt.savefig(out_psth, dpi=150)
            print("Saved:", out_psth)
        else:
            plt.show()
    else:
        print("时间范围或脉冲数为 0，跳过 PSTH。")

if __name__ == "__main__":
    main()
