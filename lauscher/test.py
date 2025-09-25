import torch
from utils.launcher_to_tensor import encode_via_launcher_to_tensor
from lauscher.spike_train import SpikeTrain

def main():
    # 指定你的 audio 文件路径
    audio_path = "test/resources/spoken_digit.flac"

    # Step 1: 使用 launcher 编码，得到稀疏事件 (2, K) tensor
    events_2xK = encode_via_launcher_to_tensor(audio_path, num_channels=70, device="cpu", verbose=True)
    print("Sparse events (2 x K):")
    print("shape:", events_2xK.shape)
    print(events_2xK[:, :10])   # 打印前10个 [time, label]

    # Step 2: 转为 SpikeTrain 对象（方便使用 to_dense）
    st = SpikeTrain.from_sparse(events_2xK.cpu().numpy())

    # Step 3: 转成稠密张量 (T, N)
    n_steps = 100     # 你 SNN 里常用的时间步
    num_channels = 70 # 和 encode 时候保持一致
    dense = st.to_dense(n_steps, num_channels, device="cpu", dtype=torch.float32)
    print("\nDense tensor (T x N):")
    print("shape:", dense.shape)
    print(dense[:10, :10])   # 打印前10个时间步 × 前10个神经元

if __name__ == "__main__":
    main()