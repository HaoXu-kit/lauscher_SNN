#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用法：
#   python dump_spikes.py my_first_spiketrain.npz --out spikes.csv

import argparse, os, numpy as np

def load_spikes(npz_path, key=None):
    npz = np.load(npz_path, allow_pickle=True)
    print("NPZ keys:", npz.files)

    # 优先用指定 key；否则猜测
    if key and key in npz.files:
        arr = np.asarray(npz[key])
    else:
        # 只有一个键时直接取
        if len(npz.files) == 1:
            arr = np.asarray(npz[npz.files[0]])
        else:
            # 常见命名
            if {'times','channels'} <= set(npz.files):
                times = np.asarray(npz['times'], dtype=float).ravel()
                chans = np.asarray(npz['channels'], dtype=int).ravel()
                return times, chans
            raise ValueError("无法确定使用哪个键，请加 --key 指定。")

    # 你的格式：2×N 或 N×2
    if arr.ndim == 2 and arr.shape[0] == 2:
        times = arr[0].astype(float).ravel()
        chans = arr[1].astype(int).ravel()
        return times, chans
    if arr.ndim == 2 and arr.shape[1] == 2:
        times = arr[:,0].astype(float).ravel()
        chans = arr[:,1].astype(int).ravel()
        return times, chans

    # 单列 times 的兜底
    if arr.ndim == 1:
        return arr.astype(float).ravel(), np.zeros(arr.size, dtype=int)

    raise ValueError(f"不支持的数组形状: {arr.shape}")

def main():
    ap = argparse.ArgumentParser(description="Dump spikes (times, channels) from NPZ.")
    ap.add_argument("npz_path", help=".npz 文件路径")
    ap.add_argument("--key", help="NPZ 内的键名（默认自动猜测）")
    ap.add_argument("--out", default="spikes.csv", help="导出的 CSV 文件名")
    args = ap.parse_args()

    if not os.path.isfile(args.npz_path):
        raise FileNotFoundError(args.npz_path)

    times, chans = load_spikes(args.npz_path, key=args.key)

    # 统计
    import numpy as np
    n = times.size
    tmin = float(np.min(times)) if n else 0.0
    tmax = float(np.max(times)) if n else 0.0
    n_ch = int(np.max(chans)+1) if n else 0
    print(f"Spikes: {n} | Channels≈{n_ch} | Time span: [{tmin:.6f}, {tmax:.6f}] s")

    # 写 CSV（不依赖 pandas）
    with open(args.out, "w") as f:
        f.write("time_s,channel\n")
        for t,c in zip(times, chans):
            f.write(f"{t},{int(c)}\n")
    print(f"已导出：{args.out}")

    # 预览前10行
    print("前10条：")
    for i in range(min(10, n)):
        print(f"{i+1:02d}  t={times[i]:.6f}s  ch={int(chans[i])}")

if __name__ == "__main__":
    main()
