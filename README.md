# $\mathrm{M}^3\text{PD}$ Dataset: Dual-view  Photoplethysmography (PPG) Using Front-and-rear Cameras of Smartphones in Lab and Clinical Settings

---

## Abstract
Portable physiological monitoring is essential for early detection and management of cardiovascular disease, but current methods often require specialized equipment that limits accessibility or impose impractical postures that patients cannot maintain. Video-based photoplethysmography on smartphones offers a convenient non-invasive alternative, yet it still faces reliability challenges caused by motion artifacts, lighting variations, and single-view constraints. Few studies have demonstrated that this technology can be reliably applied to physiological monitoring of cardiovascular patients, and no widely used open datasets exist for researchers to examine its cross-device accuracy. To address these limitations, we introduce the $\mathrm{M}^3\text{PD}$ dataset—the first publicly available dual-view mobile photoplethysmography dataset—comprising synchronized facial and fingertip videos captured simultaneously via front and rear smartphone cameras from 60 participants (including 47 cardiovascular patients). Building on this dual-view setting, we further propose the $\mathrm{F}^3\text{Mamba}$, which fuses the facial and fingertip views through Mamba-based temporal modeling. The model reduces heart-rate error by 21.9--30.2\% over existing single-view baselines while showing enhanced robustness across challenging real-world scenarios.
![F³Mamba Model Structure](Fig/dual_recording_samples.png)

## 📁 Dataset

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
