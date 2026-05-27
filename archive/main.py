import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
import config
from Process.Trainer import Trainer
from Process.data_process import MultimodalDataset
from Models.RhythmFormer import RhythmFormer
from Models.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from Models.InfinitePulse import InfinitePulse

try:
    from Models.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
except ImportError:
    PhysNet_padding_Encoder_Decoder_MAX = None

try:
    from Models.PhysMamba import PhysMamba
except ImportError:
    PhysMamba = None

try:
    from Models.F3Mamba import F3Mamba
except ImportError:
    F3Mamba = None
import wandb
import time
import os
import glob

args = config.get_config()

_dataset_cache = {}
all_dataset_ids = {
    "Lab_multimodal": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "AZDualCamera": None,  # 将在运行时动态加载
}


def load_dataloader_user_ids(train_ids=[1], test_ids=[3]):
    """
    Load and combine multiple training and testing datasets, then create DataLoaders.
    Supports three modes:
    1. Normal mode: separate train and test datasets based on user IDs.
    2. Mixed mode: combine train and test IDs and split them with given ratios.
    
    Args:
        train_ids : list of int
        test_ids : list of int
    
    Returns:
        tuple: (train_loader, test_loader, test_loader)
    """
    def _load_single_dataset(user_id, is_train=True):
        """Load a single user's dataset (with caching)."""
        dataset_name = args.dataset_name

        # 构建路径 - 支持 Lab_multimodal 和 AZDualCamera
        if dataset_name == "Lab_multimodal":
            path = fr'F3Mamba/Dataset/ProcessedDataset/{dataset_name}/front_{args.front_standard_type}_label_{args.label_standard_type}_dataset_{user_id}.pth'
        elif dataset_name == "AZDualCamera":
            path = fr'./ProcessedDataset/{dataset_name}/{dataset_name}_{args.front_standard_type}_label_{args.label_standard_type}_dataset_{user_id}.pth'
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        cache_key = (dataset_name, args.front_standard_type, args.label_standard_type, user_id)
        
        if cache_key in _dataset_cache:
            print(f"Loading from cache: {path}")
            return _dataset_cache[cache_key]

        start_time = time.time()
        dataset = torch.load(path)
        load_time = time.time() - start_time
        print(f"Loading {path}: {load_time:.2f}")

        if dataset_name == "Lab_multimodal":
            dataset = Subset(dataset, range(3, len(dataset)-1))
        elif dataset_name == "AZDualCamera":
            # AZDualCamera 数据集通常较短，去掉首尾各1个序列
            dataset = Subset(dataset, range(1, len(dataset)-1))
        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        _dataset_cache[cache_key] = dataset
        print(f"Successfully loaded {'training' if is_train else 'testing'} dataset: {path}, length: {len(dataset)}")
        return dataset

    loaded_train_datasets = [
        _load_single_dataset(train_id, is_train=True) 
        for train_id in train_ids if train_id is not None
    ]
    loaded_train_datasets = [d for d in loaded_train_datasets if d is not None] 
    
    loaded_test_datasets = [
        _load_single_dataset(test_id, is_train=False) 
        for test_id in test_ids if test_id is not None
    ]
    loaded_test_datasets = [d for d in loaded_test_datasets if d is not None]

    combined_train = ConcatDataset(loaded_train_datasets) if loaded_train_datasets else None
    combined_test = ConcatDataset(loaded_test_datasets) if loaded_test_datasets else None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True
    }
    
    train_loader = DataLoader(combined_train, shuffle=True, **loader_kwargs) if combined_train else None
    test_loader = DataLoader(combined_test, shuffle=False, **loader_kwargs) if combined_test else None

    return train_loader, test_loader, test_loader

def seed_everything(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_model(args):
    """
    Select and initialize the appropriate model based on the provided arguments.

    Args:
        args (argparse.Namespace or dict): Configuration containing model selection parameters.

    Returns:
        torch.nn.Module: The initialized model instance.
    """
    if len(args.modal_used) == 1:
        if args.video_backbone == "PhysNet_padding_Encoder_Decoder_MAX":
            model = PhysNet_padding_Encoder_Decoder_MAX(frames=args.seq_len).to(args.device)
        elif args.video_backbone == "RhythmFormer":
            model = RhythmFormer().to(args.device)
        elif args.video_backbone == "ViT_ST_ST_Compact3_TDC_gra_sharp":
            theta = 0.7
            num_heads = 4
            ff_dim = 144
            dim = 96
            patch_size = 4
            model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(args.seq_len, args.resize_height, args.resize_width),
                                                     patches=(patch_size,) * 3, dim=dim, ff_dim=ff_dim, num_heads=num_heads,
                                                     theta=theta).to(args.device)
        elif args.video_backbone == "PhysMamba":
            model = PhysMamba(theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=args.seq_len).to(args.device)
        elif args.video_backbone == "InfinitePulse":
            model = InfinitePulse(
                n_layers=[2, 2, 2, 2],
                filters=[32, 32, 32, 32],
                chunk_size=args.seq_len,
            ).to(args.device)
    else:
        if args.modal_fusion_strategy == "F3Mamba":
            model = F3Mamba(args).to(args.device)
        else:
            raise ValueError(f"Unsupported model name: {args.modal_fusion_strategy}")    
            
    return model


def get_az_dual_camera_ids():
    """动态获取 AZDualCamera 已处理的数据集 ID 列表"""
    pattern = f'./ProcessedDataset/AZDualCamera/AZDualCamera_{args.front_standard_type}_label_{args.label_standard_type}_dataset_*.pth'
    files = glob.glob(pattern)
    ids = []
    for f in files:
        # 提取文件名中的 ID: AZDualCamera_diff_normalize_label_diff_normalize_dataset_2654654_V001.pth
        basename = os.path.basename(f).replace('.pth', '')
        parts = basename.split('_dataset_')
        if len(parts) == 2:
            ids.append(parts[1])
    return sorted(ids)


def three_fold_run():
    """Run 3-fold cross-validation."""
    wandb.init(settings=wandb.Settings(init_timeout=240)) 
    print("Run")

    for key in wandb.config.keys():
        setattr(args, key, wandb.config[key])

    print(f"Start dataset: {args.dataset_name} loading!")

    # 如果是 AZDualCamera，动态获取所有已处理的 ID
    if args.dataset_name == "AZDualCamera":
        all_ids = get_az_dual_camera_ids()
        print(f"Found {len(all_ids)} processed AZDualCamera datasets")
        # 使用 fold_idx 来选择 train/test 划分
        fold_idx = args.fold_idx
        n_total = len(all_ids)
        n_test = max(1, n_total // 5)  # 20% 作为测试集
        test_ids = all_ids[fold_idx * n_test:(fold_idx + 1) * n_test]
        train_ids = [uid for uid in all_ids if uid not in test_ids]
        print(f"Fold {fold_idx}: train={len(train_ids)}, test={len(test_ids)}")
    else:
        train_ids = args.train_ids
        test_ids = args.test_ids
        
    print(f"Training IDs: {train_ids}")
    print(f"Testing IDs: {test_ids}")

    train_loader, eval_loader, test_loader = load_dataloader_user_ids(train_ids=train_ids, test_ids=test_ids)

    print("Finish data loading!")
    model = select_model(args)
    trainer = Trainer(args, model, train_loader, eval_loader, test_loader)
    trainer.train()

if __name__ == '__main__':
    seed_everything(args.seed)

    sweep_config = {
        'method': 'grid',
        'parameters': {
            'dataset_name': {
                'values': ["Lab_multimodal"]
            },
            'modal_used': {
                'values': [["front", "back"]]
            },
            'modal_fusion_strategy': {
                'values': ["F3Mamba"]
            },
            'video_backbone': {
                'values': ["PhysMamba", "RhythmFormer", "PhysNet_padding_Encoder_Decoder_MAX", "ViT_ST_ST_Compact3_TDC_gra_sharp"]
            },
            'fold_idx': {
                'values': [0, 1, 2]
            },
            'batch_size': {
                'values': [4]
            },
            'lr': {
                'values': [5e-5]
            },
            'epochs': {
                'values': [15]
            },
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=f"XXX")

    wandb.agent(sweep_id, three_fold_run)
