# $\mathrm{M}^3\text{PD}$ Dataset: Dual-view  Photoplethysmography (PPG) Using Front-and-rear Cameras of Smartphones in Lab and Clinical Settings

---

## Abstract
Portable physiological monitoring is essential for early detection and management of cardiovascular disease, but current methods often require specialized equipment that limits accessibility or impose impractical postures that patients cannot maintain. Video-based photoplethysmography on smartphones offers a convenient non-invasive alternative, yet it still faces reliability challenges caused by motion artifacts, lighting variations, and single-view constraints. Few studies have demonstrated that this technology can be reliably applied to physiological monitoring of cardiovascular patients, and no widely used open datasets exist for researchers to examine its cross-device accuracy. To address these limitations, we introduce the $\mathrm{M}^3\text{PD}$ dataset—the first publicly available dual-view mobile photoplethysmography dataset—comprising synchronized facial and fingertip videos captured simultaneously via front and rear smartphone cameras from 60 participants (including 47 cardiovascular patients). Building on this dual-view setting, we further propose the $\mathrm{F}^3\text{Mamba}$, which fuses the facial and fingertip views through Mamba-based temporal modeling. The model reduces heart-rate error by 21.9--30.2\% over existing single-view baselines while showing enhanced robustness across challenging real-world scenarios.
![F³Mamba Model Structure](Fig/dual_recording_samples.png)

## 📁 Dataset

### Datasets Comparison

Details of widely-used video physiological sensing datasets.

| Dataset | Scenarios | Subjects | Camera | Position | Vitals |
| :--- | :---: | :---: | :--- | :---: | :--- |
| PURE [1] | Lab | 10 | eco274CVGE | Face | PPG/SpO$_2$ |
| UBFC-rPPG [2] | Lab | 42 | Logitech C920 | Face | PPG |
| Oximetry [3] | Lab | 6 | Google Nexus 6P | Finger | SpO$_2$ |
| MMPD [4] | Lab | 33 | Galaxy S22 Ultra | Face | PPG |
| RLAP [5] | Lab | 58 | Logitech C930c | Face | PPG |
| SUMS [6] | Lab | 10 | Logitech C922 | Face+Finger | PPG/SpO$_2$/RR |
| LADH [7] | Lab | 21 | Logitech C922 | Face(RGB+IR) | PPG/SpO$_2$/RR |
| **$\mathrm{M}^3 \text{PD}$ (Ours)** | **Lab** | **13** | **OPPO A52** | **Face+Finger** | **PPG/SpO$_2$/RR/BP** |
| **$\mathrm{M}^3 \text{PD}$ (Ours)** | **Clinic** | **47** | **XiaoMi 14** | **Face+Finger** | **PPG/SpO$_2$/RR/BP** |

*Note: References [1]-[7] correspond to the respective datasets' origin papers.*

### $\mathrm{M}^3\text{PD}$ 
The dataset comprises synchronized physiological data from 60 participants across two collection environments:
<img src="Fig/Collection.png" width="200">

#### Lab Environment
- **Participants**: 13 healthy volunteers
- **Setting**: Controlled laboratory conditions
- **Duration**: ~15 minutes per session

#### Clinic Environment  
- **Participants**: 47 cardiovascular patients
- **Setting**: Clinical environment
- **Duration**: ~30 seconds per session
  
#### Dataset Organization

```
📁 LabDataset/
├── 📁 1/
│   ├── 📁 DualCamera_<timestamp>/
│   │   ├── 🎥 front_camera_<timestamp>.mp4           # Facial video recording
│   │   ├── 🎥 back_camera_<timestamp>.mp4            # Fingertip video recording
│   │   ├── 📊 front_camera_data_<timestamp>.txt     # Facial video timestamp data
│   │   └── 📊 back_camera_data_<timestamp>.txt      # Fingertip video timestamp data
│   └── 📁 1_spO2_rr_data/v01/
│       ├── 💓 BVP.csv                                # Ground truth Blood Volume Pulse
│       ├── ❤️ HR.csv                                 # Ground truth Heart Rate
│       ├── 🫁 RR.csv                                 # Ground truth Respiration Rate
│       ├── 🩸 SpO2.csv                               # Ground truth Blood Oxygen Saturation
│       └── ⏰ frames_timestamp.csv                   # Temporal synchronization data
├── 📁 2/
│   └── ... (similar structure)
├── ...
└── 📁 13/

```

#### Data Modalities
- **📹 Front Camera**: Facial video for remote photoplethysmography (rPPG)
- **📹 Back Camera**: Fingertip video for contact-based PPG
- **💓 Physiological Labels**: BVP, HR, RR, SpO2, Blood Pressure

#### Technical Specifications
| Modality | Specifications |
|----------|---------------|
| **Video Resolution** | 128×128 pixels |
| **Frame Rate** | 30 FPS |
| **Sequence Length** | 160 frames (5.33 seconds) |
| **Data Format** | PyTorch tensors (.pth files) |


## 🏗️ F³Mamba Architecture
The **F³Mamba** framework is designed to effectively integrate complementary physiological signals from dual-camera smartphone recordings. Our architecture leverages the power of Mamba blocks for long-range temporal dependency modeling while introducing novel fusion mechanisms for multimodal integration.

![F³Mamba Model Structure](Fig/dual_model_frame.png)

## 💻 Examples of Data Processing
### Basic Data Loading

```python
from Process.data_process import MultimodalDataLoader
import config

# Initialize configuration
args = config.get_config()

# Load Lab dataset
lab_loader = MultimodalDataLoader(config=args)
lab_loader.dataset_name = "Lab_multimodal"
lab_loader.save_datasets("./ProcessedDataset")
```
### Data Structure

```python
# Sample data structure
sample = {
    "modals": {
        "video_front": torch.Tensor,    # [seq_len, H, W, 3] - Facial video
        "video_back": torch.Tensor,     # [seq_len, H, W, 3] - Fingertip video  
    },
    "labels": {
        "bvp": torch.Tensor,           # [seq_len] - Blood Volume Pulse
        "hr": torch.Tensor,            # [seq_len] - Heart Rate
        "rr": torch.Tensor,            # [seq_len] - Respiration Rate
        "spo2": torch.Tensor,          # [seq_len] - Blood Oxygen Saturation
    }
}
```


## 💻 Examples of Network Training

### Single-Modal Training

```python
from Models.PhysMamba import PhysMamba
from Models.RhythmFormer import RhythmFormer
from Process.Trainer import Trainer

# Initialize single-modal model
args.modal_used = ["front"]  # or ["back"]
args.video_backbone = "PhysMamba"  # or "RhythmFormer", "PhysNet"

if args.video_backbone == "PhysMamba":
    model = PhysMamba(theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=args.seq_len)
elif args.video_backbone == "RhythmFormer":
    model = RhythmFormer()

# Setup training
trainer = Trainer(model, args)
trainer.train(train_loader, val_loader)
```

### Multi-Modal Fusion Training

```python
from Models.F3Mamba import F3Mamba

# Configure fusion training
args.modal_used = ["front", "back"]
args.modal_fusion_strategy = "F3Mamba"

# Initialize fusion model
model = F3Mamba(args)

# Training with multiple GPUs
trainer = Trainer(model, args)
trainer.train(train_loader, val_loader)
```

## 🗝️ Access and Usage

There are two ways for downloads： OneDrive and Baidu Netdisk. 

To access the dataset, you are supposed to download this [data release agreement TBD].  
Please scan and dispatch the completed agreement via your institutional email to <tjk24@mails.tsinghua.edu.cn> and cc <yuntaowang@tsinghua.edu.cn>. The email should have the subject line 'LADH Access Request -  your institution.' In the email,  outline your institution's **website** and **publications** for seeking access to the LADH, including its intended application in your specific research project. The email should be sent by a **faculty** rather than a student.   

## 📊 Results

### Intra-dataset Testing Results on $\mathrm{M}^3\text{PD}$

| Method | Input | Lab MAE↓ | Lab MAPE↓ | Lab RMSE↓ | Lab ρ↑ | Clinic MAE↓ | Clinic MAPE↓ | Clinic RMSE↓ | Clinic ρ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PhysNet | Face | 31.651 | 37.350 | 39.238 | -0.057 | 25.159 | 32.158 | 30.951 | 0.047 |
| PhysNet | Finger | 10.325 | 10.464 | 19.563 | 0.640 | 16.476 | 20.971 | 22.738 | 0.385 |
| PhysFormer | Face | 23.691 | 27.268 | 28.923 | 0.031 | 19.570 | 26.432 | 23.933 | 0.094 |
| PhysFormer | Finger | 16.054 | 17.242 | 24.834 | 0.363 | 13.885 | 17.384 | 17.447 | 0.350 |
| RhythmFormer | Face | 26.633 | 30.341 | 34.772 | 0.014 | 28.157 | 37.103 | 34.190 | -0.241 |
| RhythmFormer | Finger | 21.790 | 23.571 | 29.379 | 0.025 | 24.107 | 31.836 | 31.081 | -0.341 |
| PhysMamba | Face | 14.041 | 13.341 | 22.759 | 0.428 | 15.481 | 20.269 | 20.032 | 0.032 |
| PhysMamba | Finger | 9.542 | 9.247 | 18.088 | 0.630 | 9.480 | 11.411 | 15.524 | 0.460 |
| EarlyFuse-Concat| Face+Finger | 10.891 | 9.894 | 22.195 | 0.350 | 9.235 | 11.709 | 14.484 | 0.533 |
| LateFuse-Concat | Face+Finger | 15.930 | 16.820 | 20.965 | 0.045 | 14.066 | 17.443 | 17.463 | 0.152 |
| LateFuse-Avg    | Face+Finger | 7.121  | 7.208 | 13.622 | 0.598  | 8.853  | 10.940 | 14.179 | 0.539 |
| SummitVital     | Face+Finger | 6.971 | 7.168 | 13.636 | 0.612  | 9.954  | 12.645  | 15.492 | 0.460 |
| **$\mathrm{F}^3\text{Mamba}$ (Ours)** | **Face+Finger** | **6.664** | **6.859** | **12.796** | **0.636** | **7.405** | **9.308** | **10.669** | **0.753** |

### Cross-dataset Testing Results on $\mathrm{M}^3\text{PD}$

| Method | Input | Lab→Clinic MAE↓ | Lab→Clinic MAPE↓ | Lab→Clinic RMSE↓ | Lab→Clinic ρ↑ | Clinic→Lab MAE↓ | Clinic→Lab MAPE↓ | Clinic→Lab RMSE↓ | Clinic→Lab ρ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PhysNet | Face | 21.177 | 29.079 | 28.409 | 0.322 | 27.234 | 34.283 | 34.287 | -0.143 |
| PhysNet | Finger | 24.383 | 31.654 | 38.786 | 0.102 | 18.537 | 21.579 | 27.728 | 0.143 |
| PhysFormer | Face | 15.926 | 21.773 | 19.831 | 0.106 | 18.771 | 23.137 | 23.594 | 0.008 |
| PhysFormer | Finger | 14.673 | 19.173 | 19.693 | 0.099 | 15.789 | 19.238 | 22.590 | 0.120 |
| RhythmFormer | Face | 18.431 | 23.582 | 26.705 | 0.263 | 21.250 | 25.244 | 27.542 | -0.041 |
| RhythmFormer | Finger | 15.489 | 19.822 | 19.413 | -0.090 | 19.160 | 22.908 | 25.616 | -0.085 |
| PhysMamba | Face | 12.352 | 16.917 | 16.776 | 0.274 | 14.053 | 16.740 | 19.352 | 0.218 |
| PhysMamba | Finger | 8.629 | 10.840 | 12.850 | 0.599 | 8.522 | 9.302 | 15.640 | 0.523 |
| EarlyFuse-Concat| Face+Finger | 8.250 | 10.841 | 12.634 | 0.623 | 10.091 | 11.085 | 17.347 | 0.439 |
| LateFuse-Concat | Face+Finger | 14.788 | 16.857 | 19.404 | -0.184 | 16.357 | 19.126 | 21.108 | -0.046 |
| LateFuse-Avg    | Face+Finger | 8.974  | 10.757 | 16.494 | 0.358  | 9.624  | 10.011 | 18.527 | 0.337 |
| SummitVital     | Face+Finger | 10.445 | 11.627 | 18.439 | 0.409  | 8.515  | 9.451  | 16.589 | 0.575 |
| **$\mathrm{F}^3\text{Mamba}$ (Ours)** | **Face+Finger** | **8.204** | **10.115** | **12.383** | **0.644** | **9.360** | **10.938** | **15.059** | **0.546** |

### Complexity Comparison

| Method | Param (M) ↓ | FLOPs (G) ↓ | Storage (MB) ↓ | Latency (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| PhysNet | 8.85 | 70.32 | 3.38 | 16.99 |
| PhysFormer | 73.81 | 38.53 | 12.69 | 15.71 |
| RhythmFormer | 33.26 | 49.53 | 75.80 | 17.07 |
| PhysMamba | 7.59 | 60.40 | 2.90 | 23.74 |
| **$\mathrm{F}^3\text{Mamba}$ (Ours)** | **13.87** | **113.46** | **5.29** | **26.70** |

*Note: The latency is calculated using 160 frames ($160 \times 128 \times 128$) as input on a single RTX 3090 GPU.*


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tang2025dual,
  title={Dual-Camera Fusion for Robust Video-Based Photoplethysmography on Smartphones},
  author={Jiankai Tang, Tao Zhang, Jia Li, Yiru Zhang, Mingyu Zhang, Kegang Wang, Yuming Hao, Bolin Wang, Haiyang Li, Yuanchun Shi, Yuntao Wang, and Sichong Qian},
  journal={arxiv},
  year={2025}
}
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests or create issues for bugs and feature requests.

---

⭐ **Star this repo if you find it helpful!** ⭐
