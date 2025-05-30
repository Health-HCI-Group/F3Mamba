import argparse
import os
import json

def get_config():
    parser = argparse.ArgumentParser()

    # total config
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:1")

    # model config 
    parser.add_argument("--video_backbone", type=str, default="PhysMamba", choices=["RhythmFormer",
                                                                                "PhysNet_padding_Encoder_Decoder_MAX",
                                                                                "ViT_ST_ST_Compact3_TDC_gra_sharp",
                                                                                "PhysMamba",
                                                                                "PhysMamba_fft"])
    parser.add_argument("--imu_backbone",type=str,default="")
    
    parser.add_argument('--theta',type=float,default=0.5,help='weight of front and back')
    parser.add_argument('--drop_rate',type=float,default=0.5,help='drop rate of front and back')

    # fusion config
    parser.add_argument("--modal_used",type=str,default=["front"],choices=[["front"],["back"],["imu"],["front","back"],["front","back","imu"]])
    parser.add_argument("--modal_fusion_strategy",type=str,default='FusionPhysMamba_V03',choices=["FusionPhysMamba_V03"])
    parser.add_argument("--model_end_fusion_strategy",type=str,default="Concatenate",choices =["Concatenate","SummitVital"])
    parser.add_argument("--fusion_stage",type=str, default=["stage1"])
    parser.add_argument('--drop_rate_path',type=float,default=0.1,help='drop rate of front and back')


    # loss config
    parser.add_argument("--loss_type",type=str,default="time",choices=["time","frequency","time+frequency"])

    # data config
    parser.add_argument('--use_front', type=bool, default=False)
    parser.add_argument('--use_back', type=bool, default=True)
    parser.add_argument("--front_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--back_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--label_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--imu_standard_type",type=str,default="Standardized",choices=['Standardized','diff_normalize'])
    
    parser.add_argument("--dataset_name",type=str,default="tjk_multimodal",choices=["tjk_multimodal","anzhen_multimodal"])
    parser.add_argument("--data_path", type=str, default=r"/data01/tjk/多设备生理感知实验202412")
    parser.add_argument("--seq_len", type=int, default=160)
    parser.add_argument("--resize_height", type=int, default=128)
    parser.add_argument("--resize_width", type=int, default=128)
    parser.add_argument("--target_hz", type=int, default=30)

    # train config
    parser.add_argument("--fold_idx",type=int,default=0)
    parser.add_argument("--train_ids", type=list, default=[7,8,9,10,11,12,13,14,15,17,18])
    parser.add_argument("--test_ids", type=list, default=[1,3])
    parser.add_argument("--cross_validation", type=bool, default=True)
    parser.add_argument("--mixed_split", type=bool, default=False)
    parser.add_argument("--train_test_strategy", type=str, default="half->half", choices=["half->half","half->all","all->all","cross_dataset"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=int, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4,choices=[3e-3,1e-4]) # 3e-3 for mamba, 1e-4 for others
    parser.add_argument("--lr_decay_steps",type=int,default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")

    # Ablation study config
    parser.add_argument("--ablation_type",type=str,default="v0")

    # crossdatast test config
    parser.add_argument("--cross_dataset", type=str, default="tjk->anzhen")

    # test config
    # 12->64s, 6->32s
    parser.add_argument("--hr_calculate_step",type=int,default=4)

    args = parser.parse_args()
    return args
