# process_bvh_for_19_joints.py

import os
import os.path as osp
import numpy as np
import pickle
import logging
from glob import glob
import re
# 新增导入，用于处理欧拉角到旋转矩阵的转换，提高精度
from scipy.spatial.transform import Rotation as R

# -------------------------- 配置参数 --------------------------
# 核心修改：根据用户最新要求，将关节数改为19个关节
BVH_DIR = "./bvh_dataset_19"     # 输入目录名，可根据实际情况修改
OUTPUT_DIR = "./bvh_processed_19" # 输出目录名，可根据实际情况修改
TARGET_FRAMES = 64
JOINT_NUM = 19  # 核心修改：明确设置为目标19个关节
SKE_NAME_FILE = osp.join(OUTPUT_DIR, "statistics", "bvh_available_name.txt")
LABEL_SAVE_PATH = osp.join(OUTPUT_DIR, "raw_data", "labels.pkl")

EMOTION_LABEL_MAP = {
    "H": ("happiness", 0),
    "SA": ("sadness", 1),
    "N": ("neutral", 2),
    "A": ("anger", 3),
    "D": ("disgust", 4),
    "F": ("fear", 5),
    "SU": ("surprise", 6)
}
# --------------------------------------------------------------

def create_directories():
    """创建必要的输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(osp.join(OUTPUT_DIR, "raw_data"), exist_ok=True)
    os.makedirs(osp.join(OUTPUT_DIR, "statistics"), exist_ok=True)
    
    # 检查源目录是否存在.bvh文件
    bvh_files = glob(osp.join(BVH_DIR, "*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"BVH_DIR目录下未找到.bvh文件: {BVH_DIR}")
    
    # 生成有效文件名列表
    ske_names = [osp.splitext(osp.basename(f))[0] for f in bvh_files]
    np.savetxt(SKE_NAME_FILE, ske_names, fmt="%s")
    print(f"已生成有效BVH文件名列表: {SKE_NAME_FILE}")

def parse_bvh_manual(bvh_path, target_joints=19): # 核心修改：默认参数改为19
    """
    更准确地解析BVH文件，提取关节3D世界坐标序列（适配19个关节）。
    实现了基本的层级结构遍历和欧拉角旋转应用。
    """
    if os.path.getsize(bvh_path) == 0:
        print(f"  -> 警告: BVH文件为空: {bvh_path}")
        return None

    try:
        with open(bvh_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"  -> 错误: 读取文件失败: {bvh_path}, 原因: {e}")
        return None

    # --- 解析层次结构 (HIERARCHY) ---
    joints = []  # 存储关节信息 [{name, parent_idx, offset, channels}]
    joint_stack = [] # 用于追踪当前节点的栈
    joint_name_to_idx = {} # 关节名到索引的映射，方便查找父节点
    end_sites = [] # 记录End Site的数量和位置，用于最终校验
    
    hierarchy_started = False
    motion_started = False
    line_idx = 0

    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith('HIERARCHY'):
            hierarchy_started = True
            line_idx += 1
            continue
        if line.startswith('MOTION'):
            motion_started = True
            break
        if not hierarchy_started:
            line_idx += 1
            continue

        if line.startswith('ROOT') or line.startswith('JOINT'):
            joint_name = line.split()[-1]
            joint_idx = len(joints)
            parent_idx = joint_stack[-1] if joint_stack else -1 # -1 表示没有父节点 (ROOT)
            
            joints.append({
                'name': joint_name,
                'parent_idx': parent_idx,
                'offset': np.zeros(3),
                'channels': [] # 例如 ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
            })
            joint_name_to_idx[joint_name] = joint_idx
            joint_stack.append(joint_idx)
            
        elif line.startswith('End Site'):
            # End Site 通常没有名字，我们给它一个临时名字以便处理
            # 它不计入最终的 target_joints 数量，但其OFFSET对子节点位置计算有用
            parent_idx = joint_stack[-1] if joint_stack else -1
            end_site_idx = -(len(end_sites) + 1) # 用负数索引标记End Site
            joints.append({
                'name': f"EndSite_{len(end_sites)}",
                'parent_idx': parent_idx,
                'offset': np.zeros(3),
                'channels': [],
                'is_end_site': True
            })
            end_sites.append(len(joints) - 1)
            joint_stack.append(len(joints) - 1) # Push to stack for OFFSET reading
            
        elif line.startswith('OFFSET'):
            if joints:
                offset_values = list(map(float, line.split()[1:4]))
                joints[-1]['offset'] = np.array(offset_values)
                
        elif line.startswith('CHANNELS'):
            if joints:
                parts = line.split()
                num_channels = int(parts[1])
                channel_names = parts[2:2+num_channels]
                joints[-1]['channels'] = channel_names
                
        elif line.startswith('}'):
            if joint_stack:
                joint_stack.pop()
                
        line_idx += 1

    if not motion_started:
        print(f"  -> 错误: 文件中未找到 'MOTION' 关键字, 跳过: {bvh_path}")
        return None

    # 过滤掉 End Site，只保留真正的关节
    real_joints = [j for j in joints if not j.get('is_end_site', False)]
    
    # 核心修改：检查关节数是否等于目标数量 (19)
    if len(real_joints) != target_joints:
        print(f"  -> 警告: 关节数不符合要求 {target_joints} (实际: {len(real_joints)}), 跳过: {bvh_path}")
        print(f"      实际关节名: {[j['name'] for j in real_joints]}")
        return None

    # --- 解析运动数据 (MOTION) ---
    if line_idx >= len(lines):
         print(f"  -> 错误: 文件意外结束，缺少 MOTION 部分, 跳过: {bvh_path}")
         return None
         
    motion_line = lines[line_idx] # Should be 'MOTION'
    if not motion_line.startswith('MOTION'):
        print(f"  -> 错误: 无法定位到 MOTION 部分, 跳过: {bvh_path}")
        return None
        
    line_idx += 1
    if line_idx >= len(lines):
        print(f"  -> 错误: MOTION 部分不完整，缺少 Frames 或 Frame Time, 跳过: {bvh_path}")
        return None
        
    num_frames_match = re.match(r'Frames:\s*(\d+)', lines[line_idx])
    line_idx += 1
    if line_idx >= len(lines):
        print(f"  -> 错误: MOTION 部分不完整，缺少 Frame Time, 跳过: {bvh_path}")
        return None
        
    frame_time_match = re.match(r'Frame Time:\s*(\d+\.\d+)', lines[line_idx])

    if not num_frames_match or not frame_time_match:
        print(f"  -> 错误: 无法解析帧数或帧时间, 跳过: {bvh_path}")
        return None

    num_frames = int(num_frames_match.group(1))
    frame_time = float(frame_time_match.group(1))
    line_idx += 1

    # 读取所有运动数据行
    motion_data_lines = lines[line_idx:]
    
    if len(motion_data_lines) < num_frames:
        print(f"  -> 警告: 声明的帧数 ({num_frames}) 大于实际运动数据行数 ({len(motion_data_lines)})。")
        num_frames = len(motion_data_lines)
    elif len(motion_data_lines) > num_frames:
        print(f"  -> 信息: 声明的帧数 ({num_frames}) 小于实际运动数据行数 ({len(motion_data_lines)})，按声明帧数处理。")

    # 预计算每个关节在数据行中的起始通道索引
    channel_indices = []
    channel_order = [] # 记录所有通道的顺序，用于解析每一帧的数据
    idx = 0
    for j in joints: # 遍历所有节点（包括End Site），因为数据里包含它们的通道
        channel_indices.append(idx)
        channel_order.extend(j['channels'])
        idx += len(j['channels'])

    expected_values_per_frame = len(channel_order)
    # print(f"Debug: Expected values per frame: {expected_values_per_frame}")

    # 核心修改：数组形状改为 (num_frames, 19, 3)
    joints_3d = np.zeros((num_frames, target_joints, 3), dtype=np.float32)

    # --- 核心计算：逐帧计算关节世界坐标 ---
    for f in range(num_frames):
        if f >= len(motion_data_lines):
            print(f"  -> 错误: 运动数据行不足，处理到第 {f} 帧时中断。")
            break
            
        values = list(map(float, motion_data_lines[f].split()))
        
        if len(values) != expected_values_per_frame:
            print(f"  -> 警告: 第 {f} 帧数据值数量 ({len(values)}) 与预期 ({expected_values_per_frame}) 不符，跳过该帧。")
            # 用前一帧或零填充？这里选择跳过/用零填充该帧
            joints_3d[f] = 0.0
            continue

        # 存储每个关节的变换矩阵 (4x4 homogeneous transformation matrix)
        transforms = [np.eye(4) for _ in joints] 

        # 按顺序遍历所有关节（包括End Site，尽管不计算其世界坐标，但需要其OFFSET）
        for i, joint in enumerate(joints):
            parent_idx = joint['parent_idx']
            offset = joint['offset']
            channels = joint['channels']
            
            # 获取该关节对应的通道数据在values中的起始索引
            start_idx = channel_indices[i] 
            
            T_local = np.eye(4)
            T_local[:3, 3] = offset # 先应用OFFSET平移
            
            rotation = None
            position = None
            
            # 解析通道数据
            for c_idx, channel_name in enumerate(channels):
                value = values[start_idx + c_idx]
                if channel_name.endswith('rotation'):
                    if rotation is None:
                        rotation = {}
                    axis = channel_name[0].lower() # 'Xrotation' -> 'x'
                    rotation[axis] = np.radians(value) # 转为弧度
                elif channel_name.endswith('position'):
                    if position is None:
                        position = np.zeros(3)
                    axis_map = {'x': 0, 'y': 1, 'z': 2}
                    pos_axis = channel_name[0].lower()
                    if pos_axis in axis_map:
                        position[axis_map[pos_axis]] = value
            
            # 应用平移（如果是根节点或有位置通道）
            if position is not None:
                T_local[:3, 3] += position
                
            # 应用旋转
            if rotation is not None:
                # 假设旋转顺序是 channels 中出现的顺序，例如 ZXY
                # 需要根据实际的 channel 顺序来组合旋转
                axes = ''.join([c[0].lower() for c in channels if c.endswith('rotation')])
                if axes:
                    try:
                        # 使用 scipy 的 Rotation，它能很好地处理不同的顺序
                        r = R.from_euler(axes, [rotation.get(ax, 0) for ax in axes])
                        R_mat = r.as_matrix()
                        T_local[:3, :3] = R_mat
                    except ValueError as e:
                         print(f"  -> 警告: 第 {f} 帧，关节 {joint['name']} 旋转解析错误: {e}，使用单位阵。")
                         T_local[:3, :3] = np.eye(3) # 出错则不旋转
            
            # 计算世界变换矩阵
            if parent_idx == -1: # ROOT
                transforms[i] = T_local
            else:
                transforms[i] = transforms[parent_idx] @ T_local

            # 如果是真实关节（非End Site），则记录其世界坐标
            if not joint.get('is_end_site', False):
                real_joint_idx = next((idx for idx, j in enumerate(real_joints) if j['name'] == joint['name']), -1)
                if real_joint_idx != -1:
                    # 变换矩阵的最后一列（前3行）就是世界坐标
                    joints_3d[f, real_joint_idx] = transforms[i][:3, 3]

    return joints_3d


# --- 以下函数保持不变或仅微调以适应 JOINT_NUM=19 ---
# （为了完整性，这里也包含它们，但主要修改在上面的 parse_bvh_manual 和全局 JOINT_NUM）

def unify_frame_length(seq, target_length=64):
    """统一帧长度"""
    current_length = seq.shape[0]
    if current_length == target_length:
        return seq
    if current_length < target_length:
        pad_length = target_length - current_length
        last_frame = seq[-1:].repeat(pad_length, axis=0)
        return np.concatenate([seq, last_frame], axis=0)
    else:
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return seq[indices]

def normalize_sequence(seq):
    """标准化序列"""
    if seq.size == 0:
        return seq
    root_joint = seq[0, 0]  # 以第一帧根关节为原点中心化
    return seq - root_joint

def extract_emotion_label(ske_name):
    """提取情感标签"""
    match = re.match(r"^[FM](\d+)([A-Za-z]+)", ske_name)
    if not match:
        raise ValueError(f"文件名格式不匹配: {ske_name}（示例：F01A0V1、M02SU1V2）")
    emotion_abbr = match.group(2).upper()
    if emotion_abbr not in EMOTION_LABEL_MAP:
        raise ValueError(f"未知情绪缩写: {emotion_abbr}，支持的缩写：{list(EMOTION_LABEL_MAP.keys())}")
    emotion_name, label_id = EMOTION_LABEL_MAP[emotion_abbr]
    return emotion_abbr, emotion_name, label_id

def get_raw_bodies_data(bvh_path, ske_name, frames_drop_skes, frames_drop_logger):
    """获取单个BVH文件的处理后数据"""
    # 核心修改：传入19个关节的目标值
    raw_joints = parse_bvh_manual(bvh_path, target_joints=JOINT_NUM) # 使用全局 JOINT_NUM (19)
    if raw_joints is None:
        return None

    num_frames = raw_joints.shape[0]
    # 检查并过滤全零无效帧
    invalid_frames = np.where(np.all(raw_joints == 0, axis=(1, 2)))[0]
    frames_drop = invalid_frames.tolist()

    if len(invalid_frames) > 0:
        valid_mask = ~np.all(raw_joints == 0, axis=(1, 2))
        raw_joints = raw_joints[valid_mask]
        if raw_joints.shape[0] == 0:
            print(f"  -> 警告: 所有帧均为无效数据, 跳过: {ske_name}")
            return None

    # 标准化+统一帧长
    normalized_joints = normalize_sequence(raw_joints)
    unified_joints = unify_frame_length(normalized_joints, target_length=TARGET_FRAMES)

    # 构造人体数据字典（适配CTR-GCN格式）
    bodyID = "Body000"
    body_data = {
        # 核心修改：reshape 为 (64×19, 3)（帧长×关节数，展平为2D数组）
        "joints": unified_joints.reshape(-1, 3),
        # 核心修改：颜色占位数组适配19个关节（BVH无颜色信息，仅占位）
        "colors": np.zeros((TARGET_FRAMES, JOINT_NUM, 2), dtype=np.float32), # 使用全局 JOINT_NUM (19)
        "interval": list(range(TARGET_FRAMES)),  # 有效帧索引（统一为64帧）
        "motion": np.sum(np.var(unified_joints, axis=(0, 1)))  # 运动幅度计算
    }
    bodies_data = {bodyID: body_data}

    # 记录缺失帧信息
    if len(frames_drop) > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info(f"{ske_name}: 缺失{len(frames_drop)}帧（前10个索引：{frames_drop[:10]}...）")

    return {
        "name": ske_name,
        "data": bodies_data,
        "num_frames": TARGET_FRAMES
    }

def get_raw_bvh_data():
    """主处理循环"""
    ske_names = np.loadtxt(SKE_NAME_FILE, dtype=str)
    num_files = len(ske_names)
    print(f"找到{num_files}个有效BVH文件，开始处理...（目标关节数：{JOINT_NUM}）")

    raw_bvh_data, all_labels_id, all_emotion_info = [], [], []
    frames_drop_skes = dict()

    # 初始化缺失帧日志
    frames_drop_logger = logging.getLogger("bvh_frames_drop")
    frames_drop_logger.setLevel(logging.INFO)
    # 清除可能存在的旧 handlers
    for handler in frames_drop_logger.handlers[:]:
        frames_drop_logger.removeHandler(handler)
    handler = logging.FileHandler(osp.join(OUTPUT_DIR, "raw_data", "frames_drop.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) # 添加格式
    frames_drop_logger.addHandler(handler)
    # 防止重复添加handler到root logger
    frames_drop_logger.propagate = False 

    # 遍历处理每个BVH文件
    for idx, ske_name in enumerate(ske_names):
        bvh_path = osp.join(BVH_DIR, f"{ske_name}.bvh")
        print(f"\n处理 {idx + 1}/{num_files}: {osp.basename(bvh_path)}")

        # 提取情感标签
        try:
            emotion_abbr, emotion_name, label_id = extract_emotion_label(ske_name)
            print(f"  -> 情感标签: {emotion_abbr}({emotion_name}) -> ID: {label_id}")
        except ValueError as e:
            print(f"  -> 跳过: {e}")
            continue

        # 处理单个文件（提取关节数据+结构化）
        bodies_data = get_raw_bodies_data(bvh_path, ske_name, frames_drop_skes, frames_drop_logger)
        if bodies_data is None:
            print(f"  -> 跳过: 文件处理失败")
            continue

        # 记录有效样本
        raw_bvh_data.append(bodies_data)
        all_labels_id.append(label_id)
        all_emotion_info.append((emotion_abbr, emotion_name))
        print(f"  -> 成功: 生成{TARGET_FRAMES}帧数据（{JOINT_NUM}个关节）")

    # 检查是否有有效样本
    if not raw_bvh_data:
        print("\n警告: 无有效样本生成!")
        return

    # 保存结构化骨骼数据（适配CTR-GCN后续流程）
    save_data_pkl = osp.join(OUTPUT_DIR, "raw_data", "raw_bvh_data.pkl")
    with open(save_data_pkl, "wb") as f:
        pickle.dump(raw_bvh_data, f, pickle.HIGHEST_PROTOCOL)

    # 保存标签信息（含关节数记录，便于后续适配）
    label_info = {
        "all_labels_id": np.array(all_labels_id, dtype=int),  # 整数标签数组 (N,)
        "all_emotion_info": all_emotion_info,  # 情绪详细信息 (N, 2)
        "emotion_label_map": EMOTION_LABEL_MAP,  # 情绪映射表
        "num_classes": len(EMOTION_LABEL_MAP),  # 情感类别数（7类）
        "joint_num": JOINT_NUM  # 核心：记录关节数（现在是19）
    }
    with open(LABEL_SAVE_PATH, "wb") as f:
        pickle.dump(label_info, f, pickle.HIGHEST_PROTOCOL)

    # 保存有效样本帧数统计
    frames_cnt = [d["num_frames"] for d in raw_bvh_data]
    np.savetxt(
        osp.join(OUTPUT_DIR, "raw_data", "frames_cnt.txt"),
        frames_cnt,
        fmt="%d"
    )

    # 保存缺失帧信息
    frames_drop_pkl = osp.join(OUTPUT_DIR, "raw_data", "frames_drop_skes.pkl")
    with open(frames_drop_pkl, "wb") as f:
        pickle.dump(frames_drop_skes, f, pickle.HIGHEST_PROTOCOL)

    # 打印最终统计信息
    print(f"\n{'=' * 60}")
    print(f"处理完成! 关键信息汇总:")
    print(f"- 总文件数: {num_files}")
    print(f"- 有效样本数: {len(raw_bvh_data)}")
    print(f"- 关节数: {JOINT_NUM}")
    print(f"- 情感类别数: {len(EMOTION_LABEL_MAP)}")
    print(f"- 骨骼数据保存路径: {save_data_pkl}")
    print(f"- 标签信息保存路径: {LABEL_SAVE_PATH}")
    print(f"- 总有效帧数: {np.sum(frames_cnt)}")

    # 打印情感标签分布
    print("\n情感标签分布:")
    for abbr, (name, id) in EMOTION_LABEL_MAP.items():
        count = all_labels_id.count(id)
        percentage = count / len(all_labels_id) * 100 if len(all_labels_id) > 0 else 0
        print(f"  - {abbr}({name}): {count} 个样本 ({percentage:.1f}%)")

def main():
    """主函数入口"""
    try:
        create_directories()
        get_raw_bvh_data()
    except Exception as e:
        import traceback
        print(f"\n程序异常终止: {e}")
        traceback.print_exc() # 打印详细的堆栈跟踪，便于调试

if __name__ == "__main__":
    main()