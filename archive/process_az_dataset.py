"""
AZDualCamera 数据集处理脚本
============================
功能：
  1. 遍历所有被试，处理视频和标签数据
  2. 按被试ID分别保存为 .pth 文件
  3. 生成的 dataset 可直接被 main.py 中的 load_dataloader_user_ids 加载

使用方法：
  python process_az_dataset.py

输出目录：
  ./ProcessedDataset/AZDualCamera/
  文件命名格式: AZDualCamera_{front_std}_label_{label_std}_dataset_{subject_id}.pth

加载方式（在 main.py 中）：
  dataset = torch.load(path)
  # dataset 是 MultimodalDataset 对象，包含:
  #   data["video_front"]: (N, 160, 3, 128, 128) 或 None
  #   data["video_back"]: (N, 160, 3, 128, 128) 或 None
  #   data["labels"]: {
  #       "bvp": (N, 160),
  #       "hr": (N, 160),
  #       "rr": (N, 160),
  #       "spo2": (N, 160),
  #       "bp_sys": (N, 160),    # 如有
  #       "bp_dia": (N, 160),    # 如有
  #   }
"""

import sys
import os
import torch
import numpy as np

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Process.data_process import MultimodalDataLoader


class Config:
    """数据集处理配置"""
    def __init__(self):
        # === 数据集参数 ===
        self.dataset_name = "AZDualCamera"
        self.data_path = "/root/shared/az_dual_camera/az_hspt_data/data/organized"
        
        # === 视频处理参数 ===
        self.target_hz = 30          # 目标采样率 (Hz)
        self.seq_len = 160           # 序列长度 (~5.3秒 @30Hz)
        self.resize_height = 128     # 视频帧resize高度
        self.resize_width = 128      # 视频帧resize宽度
        self.use_front = True        # 使用前置摄像头 (Face.mp4)
        self.use_back = True         # 使用后置摄像头 (Finger.mp4)
        
        # === 标准化参数 ===
        # 可选: "diff_normalize" | "Standardized" | None
        self.front_standard_type = "diff_normalize"
        self.back_standard_type = "diff_normalize"
        self.label_standard_type = "diff_normalize"
        
        # === 输出路径 ===
        self.output_dir = "./ProcessedDataset/AZDualCamera"


def main():
    config = Config()
    
    print("=" * 60)
    print("AZDualCamera 数据集处理")
    print("=" * 60)
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"视频标准化: front={config.front_standard_type}, back={config.back_standard_type}")
    print(f"标签标准化: {config.label_standard_type}")
    print(f"序列长度: {config.seq_len} ({config.seq_len/config.target_hz:.1f}秒 @{config.target_hz}Hz)")
    print(f"使用摄像头: {'前置+后置' if config.use_front and config.use_back else '仅前置' if config.use_front else '仅后置'}")
    print("=" * 60)
    
    # 创建数据加载器
    data_loader = MultimodalDataLoader(config)
    
    print(f"\n共发现 {data_loader.user_number} 个有效被试会话\n")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 处理并保存每个被试
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for user_id in data_loader.user_ids:
        # 构建保存路径
        save_path = os.path.join(
            config.output_dir,
            f"AZDualCamera_{config.front_standard_type}_label_{config.label_standard_type}_dataset_{user_id}.pth"
        )
        
        # 跳过已处理的文件
        if os.path.exists(save_path):
            print(f"\n[{success_count + fail_count + skip_count + 1}/{data_loader.user_number}] ⏭️ 跳过 {user_id} (已存在)")
            skip_count += 1
            continue
        
        try:
            print(f"\n[{success_count + fail_count + skip_count + 1}/{data_loader.user_number}] 处理被试: {user_id}")
            
            # 加载并处理数据
            dataset_df = data_loader.load_merged_all_data(user_id)
            dataset = data_loader._create_dataset(dataset_df)
            
            # 保存
            torch.save(dataset, save_path, pickle_protocol=4)
            
            # 打印统计信息
            n_sequences = len(dataset)
            print(f"  ✅ 已保存: {save_path}")
            print(f"     序列数: {n_sequences}")
            
            # 打印标签信息
            if n_sequences > 0:
                sample = dataset[0]
                for label_name, label_data in sample["labels"].items():
                    print(f"     label {label_name}: shape={label_data.shape}, "
                          f"mean={label_data.mean():.4f}, std={label_data.std():.4f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"处理完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")
    print(f"数据集保存路径: {os.path.abspath(config.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
