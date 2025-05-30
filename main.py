import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset,Subset
import random
import config
from Process.Trainer import Trainer,TrainerInterable
from Process.data_process import MultimodalDataset,MultimodalIterableDataset
from Models.RhythmFormer import RhythmFormer
from Models.MultiPhysNet import PhysNet_padding_Encoder_Decoder_MAX
from Models.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from Models.Fusion import EndFusionModel 
from Models.PhysMamba import PhysMamba,FusionPhysMamba_V01,FusionPhysMamba_V02,FusionPhysMamba_V03,FusionPhysMamba_V03_1
from Models.PhysMamba_fft import PhysMambaFFT
from PIL import Image
import sys
import wandb
import time
import os
args = config.get_config()

_dataset_cache = {}
all_dataset_ids = {
    "tjk_multimodal": [1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18],
    "anzhen_multimodal": [
        '0060416775', '0060383842', '0060370217', '0060423351', '0060412868',
        '0060405974', '0060390450', '0060151006', '0053018026', '0060361762',
        '0060391976', '0060409231', '0060424393', '0060401081', '0060322775',
        '0060228059', '0060360240', '0060358129', '0060348340', '0060378992',
        '0060410660', '0053141074', '0060379119', '0060307373', '0060406606',
        '0060306867', '0060364163', '0060348962', '0060396190', '0060374625',
        '0060397284', '0060375749', '0060406882', '0060273920', '0060316427',
        '0060392842', '0060322887', '0060409023', '0060403902', '0060407838',
        '0060349954', '0060367196', '0060388433', '0060253797', '0060286108',
        '0060394791', '0060162633'
    ],
    "SUMS_multimodal": [
        "060200", "060201", "060202","060203", "060204", "060205","060206", "060207", "060208", "060209"
    ]
}

def get_cross_validation_ids(all_ids, fold_idx):
    """根据用户ID进行3折交叉验证分割"""
    # 将用户ID分为3份
    fold_size = len(all_ids) // 3
    test_start = fold_idx * fold_size
    test_end = (fold_idx + 1) * fold_size if fold_idx < 2 else len(all_ids)
    
    # 获取测试集用户ID
    test_ids = all_ids[test_start:test_end]
    # 获取训练集用户ID
    train_ids = all_ids[:test_start] + all_ids[test_end:]
    
    return train_ids, test_ids

def load_dataloader_user_ids(train_ids=[1], test_ids=[3], mixed_split=False, split_ratio=0.7, random_split=True):
    """
    Loads and combines multiple training and testing datasets from specified paths, then creates DataLoaders for them.
    Supports three modes:
    1. Normal mode: separate train and test datasets based on user ids
    2. Mixed mode: combine train and test ids and split them with a given ratios
    
    Args:
        train_ids : list of int
        test_ids : list of int
        mixed_split: bool, whether to mix train and test ids and split them
        split_ratio: float, ratio for train/test split when mixing datasets
        random_split: bool, whether to split randomly or sequentially
    
    Returns:
        tuple: (train_loader, test_loader, test_loader)
    """
    def _get_first_half(dataset):
        half_len = len(dataset) // 2
        return Subset(dataset, range(0, half_len))

    def _load_single_dataset(user_id, is_train=True):
        """加载单个用户的数据集（带缓存）"""
        # 生成唯一的缓存键（包含路径相关的所有参数）,兼容跨数据集
        if user_id in all_dataset_ids["tjk_multimodal"]:
            dataset_name = "tjk_multimodal"
        elif user_id in all_dataset_ids["anzhen_multimodal"]:
            dataset_name = "anzhen_multimodal"
        elif user_id in all_dataset_ids["SUMS_multimodal"]:
            dataset_name = "SUMS_multimodal"

        path = fr'/data01/zt/Dataset_pt/{dataset_name}/front_{args.front_standard_type}_label_{args.label_standard_type}_dataset_{user_id}.pth'
        cache_key = (dataset_name, args.front_standard_type, args.label_standard_type, user_id)
        
        # 检查缓存
        if cache_key in _dataset_cache:
            print(f"从缓存加载数据集: {path}")
            return _dataset_cache[cache_key]

        # 加载数据集
        try:
            start_time = time.time()
            dataset = torch.load(path)
            load_time = time.time() - start_time
            print(f"加载数据集 {path} 耗时: {load_time:.2f}秒")
        except FileNotFoundError:
            print(f"错误: 文件 {path} 未找到")
            return None
        except Exception as e:
            print(f"错误: 加载数据 {path} 时发生未知错误: {e}")
            return None

        # 数据集预处理（按类型分发）
        if dataset_name == "anzhen_multimodal":
            dataset = Subset(dataset, range(1, len(dataset)-1))
        elif dataset_name == "tjk_multimodal":
            # 合并重复的id处理逻辑（使用字典映射）
            special_indices = {
                9: [i for i in range(len(dataset)) if i < 100 or i > 125],
                10: [i for i in range(len(dataset)) if i != 105],
                12: [i for i in range(len(dataset)) if i != 105],
                13: [i for i in range(len(dataset)) if i not in [124,125,126]],
                15: [i for i in range(len(dataset)) if i not in [99,100,101,102]]
            }
            if user_id in special_indices:
                dataset = Subset(dataset, special_indices[user_id])
            
            # 通用处理步骤
            dataset = _get_first_half(dataset) if (dataset_half_flag['train'] if is_train else dataset_half_flag['test']) else dataset
            dataset = Subset(dataset, range(3, len(dataset)-1))
        elif args.dataset_name == "SUMS_multimodal":
            pass  # 可扩展其他预处理逻辑
        else:
            print(f"警告: 未知数据集类型 {args.dataset_name}，未进行预处理")

        # 缓存数据（仅加载成功时缓存）
        _dataset_cache[cache_key] = dataset
        print(f"成功加载{('训练' if is_train else '测试')}数据集: {path}，长度: {len(dataset)}")
        return dataset

    dataset_half_flag = {
        "train": args.train_test_strategy in ("half->half", "half->all"),
        "test": args.train_test_strategy == "half->half"
    }

    if not mixed_split:

        # 加载训练数据集
        loaded_train_datasets = [
            _load_single_dataset(train_id, is_train=True) 
            for train_id in train_ids if train_id is not None
        ]
        loaded_train_datasets = [d for d in loaded_train_datasets if d is not None]  # 过滤加载失败的
        
        # 加载测试数据集
        loaded_test_datasets = [
            _load_single_dataset(test_id, is_train=False) 
            for test_id in test_ids if test_id is not None
        ]
        loaded_test_datasets = [d for d in loaded_test_datasets if d is not None]  # 过滤加载失败的

        combined_train = ConcatDataset(loaded_train_datasets) if loaded_train_datasets else None
        combined_test = ConcatDataset(loaded_test_datasets) if loaded_test_datasets else None

    else:
        all_ids = list(set(train_ids + test_ids))
        loaded_datasets = [
            _load_single_dataset(user_id, is_train=True)  # 混合模式统一标记为训练模式预处理
            for user_id in all_ids if user_id is not None
        ]
        loaded_datasets = [d for d in loaded_datasets if d is not None]
        
        if not loaded_datasets:
            return None, None, None

        combined_dataset = ConcatDataset(loaded_datasets)
        
        if random_split:
            train_size = int(split_ratio * len(combined_dataset))
            test_size = len(combined_dataset) - train_size
            combined_train, combined_test = torch.utils.data.random_split(
                combined_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
            )
        else:
            split_idx = int(split_ratio * len(combined_dataset))
            combined_train = Subset(combined_dataset, range(split_idx))
            combined_test = Subset(combined_dataset, range(split_idx, len(combined_dataset)))
        if not loaded_datasets:
            return None, None, None

        # Combine all datasets
        combined_dataset = ConcatDataset(loaded_datasets)

        # Split dataset
        if random_split:
            # Random split
            train_size = int(split_ratio * len(combined_dataset))
            test_size = len(combined_dataset) - train_size
            combined_train_dataset, combined_test_dataset = torch.utils.data.random_split(
                combined_dataset, [train_size, test_size])
        else:
            # Sequential split
            split_idx = int(split_ratio * len(combined_dataset))
            combined_train_dataset = torch.utils.data.Subset(combined_dataset, range(0, split_idx))
            combined_test_dataset = torch.utils.data.Subset(combined_dataset, range(split_idx, len(combined_dataset)))

    combined_dataset = []

    # Create DataLoaders
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True  # 通常对GPU训练有优化
    }
    
    train_loader = DataLoader(combined_train, shuffle=True, **loader_kwargs) if combined_train else None
    test_loader = DataLoader(combined_test, shuffle=False, **loader_kwargs) if combined_test else None

    return train_loader, test_loader, test_loader

def load_dataloader_user_ids_iterable(train_ids=[1], test_ids=[3]):
    """
    Loads and combines multiple training and testing datasets from specified paths, then creates DataLoaders for them.
    Supports three modes:
    1. Normal mode: separate train and test datasets based on user ids
    2. Mixed mode: combine train and test ids and split them with a given ratios
    
    Args:
        train_ids : list of int
        test_ids : list of int

    Returns:
        tuple: (train_loader, test_loader, test_loader)
    """

    train_dataset_paths = [
        fr'/data01/zt/Dataset_pt/Iterable/{args.dataset_name}/{train_id}' 
        for train_id in train_ids
    ]

    test_dataset_paths = [
        fr'/data01/zt/Dataset_pt/Iterable/{args.dataset_name}/{test_id}' 
        for test_id in test_ids
    ]
    
    train_dataset = MultimodalIterableDataset(train_dataset_paths)
    test_dataset = MultimodalIterableDataset(test_dataset_paths)

    # Create DataLoaders
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        # shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=2,
        # shuffle=False
    ) 

    return train_loader, test_loader, test_loader

def plot_label_bvp():
    dataset = torch.load(r"/data01/zt/Dataset_pt/front_diff_normalize_label_diff_normalize_dataset_0060253797.pth")
    # 画出标签
    plt.figure(figsize=(10, 6))
    data = dataset.data['labels']['bvp'].flatten()
    plt.plot(data)
    plt.title('Label')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(f"/data01/zt/Dataset_pt/anzhen_bvp_plot/label_0060253797.png")

def visualize_dataset_images(dataset, show_grid=True):
    dataset = torch.load(r"/data01/zt/Dataset_pt/front_diff_normalize_label_diff_normalize_dataset_0060391976.pth")
    # 同时显示160张图的代码
    if show_grid:
        for k in range(len(dataset)):
            # Step: Visualize the data [160,H,W,C] 
            data_sample = dataset[k]
            video_length = data_sample["modals"]["video_front"][0].shape[0]
            fig, axes = plt.subplots(10, 16, figsize=(16, 10))
            for i in range(10):
                for j in range(16):
                    index = i * 16 + j
                    if index < video_length:
                        image = data_sample["modals"]["video_front"][index]  # 假设数据集中每个样本的第一个元素是图像
                        if isinstance(image, torch.Tensor):
                            image = image.numpy()  # 如果是Tensor，转换为numpy数组并调整维度
                        # print(image)
                        axes[i, j].imshow(image)
                        axes[i, j].axis('off')
                    else:
                        axes[i, j].axis('off')
            plt.savefig(f"/data01/zt/Dataset_pt/anzhen_front_plot/dataset_images{k}.png")

def visualize_bvp_acc_signals(dataset):
    dataset = torch.load(r"/data01/zt/Dataset_pt/front_diff_normalize_label_diff_normalize_dataset_0060253797.pth")
    # 画出标签
    plt.figure(figsize=(10, 6))
    bvp = dataset.data['labels']['bvp'].flatten()
    plt.plot(bvp)

    # imu_acc [B,L,C]->[L*B,C]
    imu_acc = dataset.data['modals']['imu_acc'].reshape(-1, dataset.data['modals']['imu_acc'].shape[-1])
    plt.plot(imu_acc)

    plt.title('Label')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig(f"/data01/zt/Dataset_pt/anzhen_bvp_plot/label_0060253797.png")

def visualize_combined():

    userid_list = ["060200"]

    # dataset_paths = [
    #         fr'/data01/zt/Dataset_pt/anzhen_multimodal/front_diff_normalize_label_diff_normalize_dataset_{user_id}.pth' 
    #         for user_id in userid_list
    #     ]
    dataset_paths = [
            fr'/data01/zt/Dataset_pt/SUMS_multimodal/front_diff_normalize_label_diff_normalize_dataset_{user_id}.pth' 
            for user_id in userid_list
        ]

    for idx, user in enumerate(userid_list):
        # Load dataset
        dataset = torch.load(dataset_paths[idx])
        indices = [i for i in range(len(dataset)) if i < 100]
        dataset = torch.utils.data.Subset(dataset, indices)

        # Create output directory if it doesn't exist
        output_dir = f"/data01/zt/Dataset_pt/SUMS_plot/{user}"
        os.makedirs(output_dir, exist_ok=True)

        for k in range(len(dataset)):
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            grid = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
            data_sample = dataset[k]

            # Video frames subplot
            ax_images = plt.subplot(grid[0])
            ax_images.set_title('Video Frames')
            ax_images.axis('off')  # Hide main axis, we'll use sub-axes

            video_length = data_sample["modals"]["video_front"][0].shape[0]
            
            # Create grid for video frames
            inner_grid = plt.GridSpec(10, 16, hspace=0.1, wspace=0.1)
            for i in range(10):
                for j in range(16):
                    index = i * 16 + j
                    ax = plt.subplot(inner_grid[i, j])
                    if index < video_length:
                        image = data_sample["modals"]["video_front"][index]
                        if isinstance(image, torch.Tensor):
                            image = image.numpy()
                        ax.imshow(image)
                    ax.axis('off')

            # BVP plot
            ax_bvp = plt.subplot(grid[1])
            bvp = data_sample['labels']['bvp']
            ax_bvp.plot(bvp)
            ax_bvp.set_title('BVP Signal')
            ax_bvp.set_xlabel('Time')
            ax_bvp.set_ylabel('Value')
            
            # IMU plot
            # ax_imu = plt.subplot(grid[2])
            # imu_acc = data_sample['modals']['imu_acc']
            # ax_imu.plot(imu_acc[:, 0], label='X-axis')
            # ax_imu.plot(imu_acc[:, 1], label='Y-axis')
            # ax_imu.plot(imu_acc[:, 2], label='Z-axis')
            # ax_imu.set_title('IMU Acceleration')
            # ax_imu.set_xlabel('Time')
            # ax_imu.set_ylabel('Value')
            # ax_imu.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dataset_images{k}.png")
            plt.close(fig)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_model(args):
    """
    Selects and initializes the appropriate model based on the provided arguments.

    Args:
        args (argparse.Namespace or dict): Command-line arguments or configuration dictionary 
            containing model selection parameters (e.g., model name, architecture).

    Returns:
        torch.nn.Module: The initialized model instance.
    """
    if len(args.modal_used)==1:
        # load model
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
            model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(args.seq_len,args.resize_height,args.resize_width),
                                                    patches =(patch_size,) * 3,dim = dim,ff_dim = ff_dim,num_heads=num_heads,
                                                    theta = theta).to(args.device)
        elif args.video_backbone == "PhysMamba":
            model = PhysMamba(theta=0.5,drop_rate1=0.25,drop_rate2=0.5,frames=args.seq_len).to(args.device)
        elif args.video_backbone == "PhysMamba_fft":
            model = PhysMambaFFT(args=args).to(args.device)

    else:
        if args.modal_fusion_strategy == "EndFusionModel":
            model = EndFusionModel(args).to(args.device)
        elif args.modal_fusion_strategy == "FusionPhysMamba_V01":
            model = FusionPhysMamba_V01(args).to(args.device)
        elif args.modal_fusion_strategy == "FusionPhysMamba_V02":
            model = FusionPhysMamba_V02(args).to(args.device)
        elif args.modal_fusion_strategy == "FusionPhysMamba_V03":
            model = FusionPhysMamba_V03(args).to(args.device)
        elif args.modal_fusion_strategy == "FusionPhysMamba_V03_1":
            model = FusionPhysMamba_V03_1(args).to(args.device)
        elif args.modal_fusion_strategy == "FusionPhysMamba_V04":
            model = FusionPhysMamba_V04(args).to(args.device)
        else:
            raise ValueError(f"Unsupported model name: {args.modal_fusion_strategy}")    
            
    return model

def three_fold_run():
    wandb.init(settings=wandb.Settings(init_timeout=240)) 
    print("run")

    for key in wandb.config.keys():
        setattr(args, key, wandb.config[key])

    print(args)

    print(f"Start dataset: {args.dataset_name} loading!")
    
    if args.cross_validation == True:
        all_ids = all_dataset_ids[args.dataset_name]
        train_ids, test_ids = get_cross_validation_ids(all_ids=all_ids,fold_idx=args.fold_idx)
        args.train_ids = train_ids
        args.test_ids = test_ids
        mixed_split = False

    else:
        train_ids = args.train_ids
        test_ids = args.test_ids
        mixed_split = args.mixed_split
        
    print(f"Training IDs: {train_ids}")
    print(f"Testing IDs: {test_ids}")

    train_loader, eval_loader, test_loader = load_dataloader_user_ids(train_ids=train_ids,test_ids=test_ids,mixed_split=mixed_split)
    # train_loader, eval_loader, test_loader = load_dataloader_user_ids_iterable(train_ids=train_ids,test_ids=test_ids)

    # if train_loader and train_loader.dataset:
    #     visualize_dataset_images(train_loader.dataset, "Training Dataset Images")
    # if test_loader and test_loader.dataset:
    #     visualize_dataset_images(test_loader.dataset, "Testing Dataset Images")

    print("Finish data loading!")

    model = select_model(args)
    trainer = Trainer(args, model, train_loader, eval_loader, test_loader)
    trainer.train()

def cross_dataset_train_test():

    wandb.init(settings=wandb.Settings(init_timeout=240)) 
    print("run")
    seed_everything(args.seed)

    for key in wandb.config.keys():
        setattr(args, key, wandb.config[key])

    print(args)

    if args.cross_dataset == "tjk->anzhen":
        train_ids = all_dataset_ids["tjk_multimodal"]
        test_ids = all_dataset_ids["anzhen_multimodal"]
        args.dataset_name = "anzhen_multimodal"

    elif args.cross_dataset == "tjk->SUMS":
        train_ids = all_dataset_ids["tjk_multimodal"]
        test_ids = all_dataset_ids["SUMS_multimodal"]
        args.dataset_name = "SUMS_multimodal"

    elif args.cross_dataset == "anzhen->tjk":
        train_ids = all_dataset_ids["anzhen_multimodal"]
        test_ids = all_dataset_ids["tjk_multimodal"]
        args.dataset_name = "tjk_multimodal"

    elif args.cross_dataset == "anzhen->SUMS":
        train_ids = all_dataset_ids["anzhen_multimodal"]
        test_ids = all_dataset_ids["SUMS_multimodal"]
        args.dataset_name = "SUMS_multimodal"

    elif args.cross_dataset == "SUMS->tjk":
        train_ids = all_dataset_ids["SUMS_multimodal"]
        test_ids = all_dataset_ids["tjk_multimodal"]
        args.dataset_name = "tjk_multimodal"

    elif args.cross_dataset == "SUMS->anzhen":
        train_ids = all_dataset_ids["SUMS_multimodal"]
        test_ids = all_dataset_ids["anzhen_multimodal"]
        args.dataset_name = "anzhen_multimodal"

    else:
        raise ValueError(f"Unsupported cross_dataset option: {args.cross_dataset}")

    # dataset loading
    train_loader, test_loader, test_loader = load_dataloader_user_ids(train_ids=train_ids,test_ids=test_ids,mixed_split=False)
    model = select_model(args)
    trainer = Trainer(args, model, train_loader, test_loader, test_loader)
    trainer.train()


# TODO tjk and SUMS 重新跑
if __name__ == '__main__':

    print(f"CUDA 版本: {torch.version.cuda}")    
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}") 
    print(f"PyTorch 版本: {torch.__version__}")  

    print(args)
    print(sys.version)
    seed_everything(args.seed)

    # train和test的组合,[half->half],[half->all],[all->all]
    # cross_dataset [tjk->anzhen],[tjk->SUMS],[anzhen->tjk],[anzhen->SUMS],[SUMS->tjk],[SUMS->anzhen]

    # TODO use the anzhen dataset ablation study
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'dataset_name': {
                'values': ["anzhen_multimodal"]
            },
            'modal_used': {
                'values': [["front","back"]]
            },
            'modal_fusion_strategy':{
                'values':["FusionPhysMamba_V03"]
            },
            # 'model_end_fusion_strategy':{
            #     'values':["MambaFusionNet"]
            # }
            # 'video_backbone':{
            #     'values':["PhysMamba","RhythmFormer","PhysNet_padding_Encoder_Decoder_MAX","ViT_ST_ST_Compact3_TDC_gra_sharp",]
            # },
            'video_backbone':{
                'values':["PhysMamba"]
            },
            "train_test_strategy":{
                'values':["all->all"]
            },
            'fold_idx':{
                'values':[0,1,2]
            },
            'batch_size':{
                'values':[4]
            },
            'lr':{
                'values':[5e-5]
            },
            'epochs':{
                'values':[15]
            },
            'dropout_rate':{
                'values':[0.3]
            },
            'drop_rate_path':{
                'values':[0.]
            },
            'theta':{
                'values':[0.3]
            },
            'fusion_stage':{
                'values':[["stage1","stage2"]]
            },
            # 'cross_dataset':{
            #     'values': ["tjk->anzhen", "tjk->SUMS", "anzhen->tjk", "anzhen->SUMS", "SUMS->tjk", "SUMS->anzhen"]
            # }

        }
    }
    sweep_id = wandb.sweep(sweep_config, project=f"PCG-anzhendataset")

    # wandb.agent(sweep_id,cross_dataset_train_test)
    wandb.agent(sweep_id, three_fold_run)   
