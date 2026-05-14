import os
import os.path as osp
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# --- 配置参数 ---
# 输入文件路径 (由您的 BVH 解析脚本生成)
RAW_DATA_PATH = "./bvh_processed_15/raw_data/raw_bvh_data.pkl"
LABEL_PATH = "./bvh_processed_15/raw_data/labels.pkl"

# 输出文件路径
SAVE_PATH = "./bvh_processed_15/final_data"
os.makedirs(SAVE_PATH, exist_ok=True)

# 训练集/测试集划分比例
TEST_SIZE = 0.2
RANDOM_STATE = 42  # 保证划分结果可复现

# 统一的目标帧长 (建议与您解析时使用的 TARGET_FRAMES 一致)
TARGET_FRAMES = 64

# --- 主函数 ---
def main():
    print("--- 开始直接标准化和数据集划分 ---")

    # 1. 加载原始数据和标签
    print(f"加载数据 from: {RAW_DATA_PATH}")
    if not osp.exists(RAW_DATA_PATH) or not osp.exists(LABEL_PATH):
        print(f"错误: 请先运行 BVH 解析脚本，确保 {RAW_DATA_PATH} 和 {LABEL_PATH} 存在。")
        return

    with open(RAW_DATA_PATH, 'rb') as f:
        raw_skes_data = pickle.load(f)

    with open(LABEL_PATH, 'rb') as f:
        label_info = pickle.load(f)

    all_labels = label_info["all_labels_id"]
    num_classes = label_info["num_classes"]
    joint_num = label_info["joint_num"]
    print(f"数据加载完成. 样本数: {len(raw_skes_data)}, 类别数: {num_classes}, 关节数: {joint_num}")

    # 2. 提取关节数据并进行初步处理
    processed_sequences = []
    valid_labels = []

    for i, bodies_data in enumerate(raw_skes_data):
        # 提取第一个人体 (Body000) 的关节数据
        # 假设你的数据都是单人动作
        body_data = list(bodies_data['data'].values())[0]
        joints_2d = body_data['joints']  # 形状: (num_frames * joint_num, 3)

        # 重塑为 (num_frames, joint_num * 3)
        # 每个帧是一个长向量，包含所有关节的 x, y, z 坐标
        num_frames_original = bodies_data['num_frames']
        joints_sequence = joints_2d.reshape(num_frames_original, joint_num * 3)

        processed_sequences.append(joints_sequence)
        valid_labels.append(all_labels[i])

    # 3. 序列对齐 (统一帧长) 和 数据标准化
    normalized_sequences = []
    print(f"开始序列对齐 (目标帧长: {TARGET_FRAMES}) 和 标准化...")

    for seq in processed_sequences:
        num_frames = seq.shape[0]

        # --- 步骤 3.1: 序列对齐 ---
        if num_frames < TARGET_FRAMES:
            # 如果帧长不足，在末尾补零
            pad_length = TARGET_FRAMES - num_frames
            padded_seq = np.pad(seq, ((0, pad_length), (0, 0)), 'constant')
            aligned_seq = padded_seq
        elif num_frames > TARGET_FRAMES:
            # 如果帧长超出，使用均匀采样的方式截取到目标长度
            indices = np.linspace(0, num_frames - 1, TARGET_FRAMES, dtype=int)
            aligned_seq = seq[indices]
        else:
            aligned_seq = seq

        # --- 步骤 3.2: 数据标准化 ---
        # 将序列重塑为 (TARGET_FRAMES, joint_num, 3) 以便进行坐标运算
        seq_3d = aligned_seq.reshape(TARGET_FRAMES, joint_num, 3)

        # a. 中心化 (Translation Normalization)
        # 找到第一帧的根关节 (假设是第 0 个关节)
        root_joint_first_frame = seq_3d[0, 0, :]
        # 所有帧的所有关节都减去根关节的初始位置
        seq_centered = seq_3d - root_joint_first_frame

        # b. 尺度归一化 (Scale Normalization)
        # 计算整个序列的骨骼包围盒对角线长度作为尺度因子
        all_joints = seq_centered.reshape(-1, 3)
        min_coords = np.min(all_joints, axis=0)
        max_coords = np.max(all_joints, axis=0)
        scale_factor = np.linalg.norm(max_coords - min_coords)

        # 避免除以零
        if scale_factor < 1e-6:
            scale_factor = 1.0

        seq_normalized_3d = seq_centered / scale_factor

        # 将标准化后的序列重塑回 (TARGET_FRAMES, joint_num * 3)
        seq_normalized = seq_normalized_3d.reshape(TARGET_FRAMES, joint_num * 3)

        normalized_sequences.append(seq_normalized)

    print("标准化完成.")

    # 4. 转换为 NumPy 数组
    X = np.array(normalized_sequences, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.int64)

    print(f"数据形状: X={X.shape}, y={y.shape}")

    # 5. 划分训练集和测试集
    print(f"划分训练集和测试集 (测试集比例: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. 保存为 .npz 文件
    train_save_path = osp.join(SAVE_PATH, "bvh_emotion_train.npz")
    test_save_path = osp.join(SAVE_PATH, "bvh_emotion_test.npz")

    np.savez(train_save_path, x=X_train, y=y_train)
    np.savez(test_save_path, x=X_test, y=y_test)

    print("--- 处理完成 ---")
    print(f"训练集已保存到: {train_save_path}")
    print(f"测试集已保存到: {test_save_path}")
    print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

if __name__ == "__main__":
    main()