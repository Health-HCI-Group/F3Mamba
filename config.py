import argparse

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
    
    parser.add_argument('--theta',type=float,default=0.5,help='weight of front and back')
    parser.add_argument('--drop_rate',type=float,default=0.5,help='drop rate of front and back')

    # fusion config
    parser.add_argument("--modal_used",type=str,default=["front"],choices=[["front"],["back"],["front","back"]])
    parser.add_argument("--modal_fusion_strategy",type=str,default='F3Mamba',choices=["F3Mamba"])
    parser.add_argument('--drop_rate_path',type=float,default=0.1,help='drop rate of front and back')

    # data config
    parser.add_argument('--use_front', type=bool, default=False)
    parser.add_argument('--use_back', type=bool, default=True)
    parser.add_argument("--front_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--back_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--label_standard_type",type=str,default="diff_normalize",choices=['Standardized','diff_normalize'])
    parser.add_argument("--imu_standard_type",type=str,default="Standardized",choices=['Standardized','diff_normalize'])
    
    parser.add_argument("--dataset_name",type=str,default="Lab_multimodal")
    parser.add_argument("--data_path", type=str, default=r"/your/data/path", help="Path to the dataset.")
    parser.add_argument("--seq_len", type=int, default=160)
    parser.add_argument("--resize_height", type=int, default=128)
    parser.add_argument("--resize_width", type=int, default=128)
    parser.add_argument("--target_hz", type=int, default=30)

    # train config
    parser.add_argument("--fold_idx",type=int,default=0)
    parser.add_argument("--train_ids", type=list, default=[7,8,9,10,11,12,13,14,15,17,18])
    parser.add_argument("--test_ids", type=list, default=[1,3])
    parser.add_argument("--cross_validation", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=int, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4,choices=[3e-3,1e-4]) # 3e-3 for mamba, 1e-4 for others
    parser.add_argument("--lr_decay_steps",type=int,default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")

    # Mamba fusion block config
    parser.add_argument("--mamba_type",type=str,default="v2")

    args = parser.parse_args()
    return args
