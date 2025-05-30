import gc
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import sys
import time
import cv2
import glob
from tqdm import tqdm
import torch
import queue
from queue import Queue
import alphashape
import threading
import mediapipe as mp
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import IterableDataset
from PIL import Image
import json
from PyEMD import CEEMDAN
from scipy.signal import fftconvolve, butter, filtfilt
from functools import lru_cache
try:
    Image.ANTIALIAS = Image.LANCZOS
except AttributeError:
    pass

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import config


class ImuAccProcessor(object):

    @staticmethod
    def soft_denoise(signal):
        """小波软阈值去噪"""
        median_value = np.median(signal)
        theta = np.median(abs(signal-median_value))/2.5
        threshold = theta * np.sqrt(np.log(2*len(signal)))
        return threshold
    
    @staticmethod
    def iceemdan_denoise(signal, num_imfs=5, noise_strength=0.2):
        """CEEMDAN/ICEEMDAN去噪"""
        ceemdan = CEEMDAN()
        imfs = ceemdan(signal, max_imf=num_imfs)
        return imfs
    
    @staticmethod
    def brick_wall_fft_filter(s, fs, lowcut, highcut):
        """使用FFT进行砖墙式带通滤波"""
        n = len(s)
        fft_s = np.fft.fft(s)
        freq = np.fft.fftfreq(n, d=1/fs)
        mask = (np.abs(freq) >= lowcut) & (np.abs(freq) <= highcut)
        fft_s_filtered = fft_s * mask
        filtered_s = np.fft.ifft(fft_s_filtered)
        return np.real(filtered_s)
    
    @staticmethod
    def remove_respiration(s, window_length=50):
        """使用均值滤波器去除呼吸成分"""
        window = np.ones(window_length) / window_length
        s_b = fftconvolve(s, window, mode='same')
        return s - s_b
    
    @staticmethod
    def butter_bandpass_filter(s, fs, lowcut, highcut, order=4):
        """巴特沃斯带通滤波器"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        nyq = 0.5 * fs
        # 计算最小所需信号长度
        padlen = 3 * max(len(b), len(a)) - 1  # filtfilt默认padlen
        min_length = padlen + 1
        
        if len(s) < min_length:
            print(f"Warning: Signal length {len(s)} < {min_length}, skipping filtering.")
            return s  # 返回原始信号或处理后的信号
        return filtfilt(b, a, s)
    
    @staticmethod
    def process_signal(s, fs=400):
        """Improved CEEMDAN-based signal processing pipeline.

        Args:
            s (np.ndarray): Input signal.
            fs (float, optional): Sampling frequency. Defaults to 400.

        Returns:
            tuple: Contains:
                - s_baseline_removed (np.ndarray): Signal after baseline removal.
                - s_combined (np.ndarray): Combined relevant IMFs.
                - s_clean (np.ndarray): Final denoised signal.
        """
        # Step 1: Baseline removal (high-pass filtering)
        s_baseline_removed = ImuAccProcessor.butter_bandpass_filter(s, fs, lowcut=0.5, highcut=13)
        
        # Step 2: CEEMDAN decomposition
        ceemdan = CEEMDAN()
        imfs = ceemdan(s_baseline_removed, max_imf=4)
        
        # Step 3: Select BCG-related IMFs
        selected_imfs = imfs[:2]
        s_combined = np.sum(selected_imfs, axis=0)
        
        # Step 4: Wavelet soft-threshold denoising
        threshold = ImuAccProcessor.soft_denoise(s_combined)
        s_clean = np.sign(s_combined) * np.maximum(np.abs(s_combined) - threshold, 0)

        return s_baseline_removed, s_combined, s_clean

             
def split_string_number(num_str):
    if len(num_str) <= 10:
        return float(num_str)
    integer_part = num_str[:10]
    decimal_part = num_str[10:]
    result = float(f"{integer_part}.{decimal_part}")
    return result


def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


def find_and_remove_continuous_values(all_data, user_id, target_value, min_length=100):
    """
    Find and remove continuous occurrences of a target value in the 'bvp' column of the DataFrame.

    Parameters:
    - all_data: DataFrame containing the 'bvp' column.
    - user_id: Identifier for the user.
    - target_value: The integer value to look for continuous occurrences.
    - min_length: The minimum length of continuous occurrences to consider (default is 100).

    Returns:
    - all_data
    """
    # Check if the 'bvp' column contains the target value
    is_target = all_data['bvp'] == target_value

    # Group continuous True values
    groups = (is_target != is_target.shift()).cumsum()

    # Filter groups where the length of continuous True values is greater than or equal to min_length
    long_groups = groups[is_target].value_counts()[lambda x: x >= min_length].index

    # Store the start and end indices of continuous target values
    start_end_indices = []
    for group in long_groups:
        group_indices = all_data.index[groups == group]
        start_index = group_indices.min()
        end_index = group_indices.max()
        start_end_indices.append((start_index, end_index))

    # Output the results and remove the continuous target values from the DataFrame
    if start_end_indices:
        print(f"{user_id}: 存在连续 {min_length} 个以上值为 {target_value} 的情况：")
        for start, end in start_end_indices:
            print(f"开头索引: {start}, 结尾索引: {end}")
            all_data.drop(all_data.index[start:end + 1], inplace=True)
    else:
        print(f"不存在连续 {min_length} 个以上值为 {target_value} 的情况。")
    
    return all_data


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, use_front=True, use_back=True):
        self.data = {
            "imu_acc": features["imu_acc"] if "imu_acc" in features else None,
            "imu_gyro": features["imu_gyro"] if "imu_gyro" in features else None,
            "video_front": features["video_front"] if "video_front" in features else None,
            "video_back": features["video_back"] if "video_back" in features else None,
            "labels": labels
        }

        self.data = {k: v for k, v in self.data.items() if v is not None}

    def __len__(self):
        return len(self.data["labels"]['bvp'])

    def __getitem__(self, idx):
        sample = {
            "modals": {},
            "labels": {}
        }
        
        # Add available modalities to sample
        for modality in ["imu_acc", "imu_gyro", "video_front", "video_back"]:
            if modality in self.data:
                sample["modals"][modality] = self.data[modality][idx]
                # Add available modalities to sample

        for modality in ["bvp", "hr", "rr", "spo2"]:
            if modality in self.data["labels"]:
                sample["labels"][modality] = self.data["labels"][modality][idx]

        return sample


class MultimodalIterableDataset(IterableDataset):
    def __init__(self, subject_dirs, use_front=True, use_back=True):
        """
        Args:
            subject_dirs: List[str] 被试文件夹路径列表
            use_front: 是否使用前视视频
            use_back: 是否使用后视视频
        """
        super().__init__()
        self.subject_dirs = subject_dirs
        self.use_front = use_front
        self.use_back = use_back
        
        # 预加载元数据（轻量）
        self.metadata_list = []
        for dir_path in self.subject_dirs:
            metadata = torch.load(os.path.join(dir_path, "metadata.pt"))
            self.metadata_list.append({
                "dir": dir_path,
                "total_samples": metadata["total_samples"],
                "modalities": metadata["modalities"],
                "label_types": metadata["label_types"]
            })

    @lru_cache(maxsize=5)  # 缓存最近两个subject的视频数据
    def _load_video_file(self, dir_path, modal):
        return torch.load(os.path.join(dir_path, f"{modal}.pt"))

    def _load_sample(self, subject_idx, sample_idx):
        """动态加载单个样本"""
        metadata = self.metadata_list[subject_idx]
        sample = {"modals": {}, "labels": {}}
        
        # 加载视频
        if self.use_front and "video_front" in metadata["modalities"]:
            dir_path = self.subject_dirs[subject_idx]
            sample["modals"]["video_front"] = self._load_video_file(dir_path, "video_front")[sample_idx]
        
        if self.use_back and "video_back" in metadata["modalities"]:
            dir_path = self.subject_dirs[subject_idx]
            sample["modals"]["video_back"] = self._load_video_file(dir_path, "video_back")[sample_idx]
        

        # 加载标签
        labels = torch.load(os.path.join(metadata["dir"], "labels.pt"))
        for label_type in metadata["label_types"]:
            sample["labels"][label_type] = labels[label_type][sample_idx]

        return sample

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        
        # 分片策略：每个worker处理固定被试
        subjects_per_worker = math.ceil(len(self.subject_dirs) / num_workers)
        start = worker_id * subjects_per_worker
        end = min(start + subjects_per_worker, len(self.subject_dirs))
        
        # 流式生成样本
        for subj_idx in range(start, end):
            total_samples = self.metadata_list[subj_idx]["total_samples"]
            for sample_idx in range(total_samples):
                yield self._load_sample(subj_idx, sample_idx)

    def __len__(self):
        return sum(m["total_samples"] for m in self.metadata_list)


class MemoryMappedMultiModalDataset(IterableDataset):
    def __init__(self, subject_dirs, buffer_size=5000):
        self.subject_dirs = subject_dirs
        self.buffer_size = buffer_size
        
        # 预加载轻量元数据
        self.metadata = [
            torch.load(os.path.join(d, "metadata.pt"), map_location="cpu")
            for d in subject_dirs
        ]
        
        # 内存映射缓存（延迟加载）
        self._mmap_cache = {}
        
    def _get_mmap(self, path):
        """获取内存映射对象（带缓存）"""
        if path not in self._mmap_cache:
            # 转换为numpy格式（预处理步骤需提前完成）
            npy_path = path.replace(".pt", ".npy")
            if not os.path.exists(npy_path):
                pt_data = torch.load(path, map_location="cpu").numpy()
                np.save(npy_path, pt_data)
            self._mmap_cache[path] = np.load(npy_path, mmap_mode='r')
        return self._mmap_cache[path]

    def __iter__(self):
        # 分片逻辑保持不变...
        for subj_idx in assigned_subjects:
            dir_path = self.subject_dirs[subj_idx]
            metadata = self.metadata[subj_idx]
            
            # 加载内存映射数据
            imu_acc = self._get_mmap(os.path.join(dir_path, "imu_acc.pt"))
            imu_gyro = self._get_mmap(os.path.join(dir_path, "imu_gyro.pt"))
            video_front = self._get_mmap(os.path.join(dir_path, "video_front.pt"))
            labels = self._get_mmap(os.path.join(dir_path, "labels.pt"))
            
            # 流式生成带缓冲shuffle
            buffer = []
            for sample_idx in range(metadata["total_samples"]):
                buffer.append({
                    "imu_acc": imu_acc[sample_idx],
                    "video_front": video_front[sample_idx],
                    "labels": labels[sample_idx]
                })
                if len(buffer) >= self.buffer_size:
                    np.random.shuffle(buffer)
                    while len(buffer) > self.buffer_size//2:
                        yield buffer.pop(0)
            # 处理剩余样本
            np.random.shuffle(buffer)
            yield from buffer


class MultimodalDataLoader():
    def __init__(self, config):
        self.dataset_name = "tjk_multimodal"
        print(f"dataset_name:{self.dataset_name}")
        self.raw_data_path = config.data_path
        self.use_front = True
        self.use_back = True

        # standard config 
        self.front_standard_type = config.front_standard_type if self.use_front else None
        self.back_standard_type  = config.back_standard_type if self.use_back else None
        self.imu_standard_type = config.imu_standard_type   
        self.label_standard_type = config.label_standard_type
        
        # resize config
        self.target_height = config.resize_height
        self.target_weight = config.resize_width
        self.target_Hz = config.target_hz
        self.seq_len = config.seq_len

        # test config
        self.test_size = config.test_size
        self.batch_size = config.batch_size
        self.dirs = self.get_user_data_dirs()
        self.labels = ["bvp","hr","rr","spo2"]
        self.user_number = len(self.dirs.keys())
        self.user_ids = self.dirs.keys()

        # face detector config
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.omitted_frames = 1
        self.buffer_size = 100
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.result_queue = Queue(maxsize=self.buffer_size)

    def get_user_data_dirs(self):
        """
        Collects data directories and relevant file paths for each user.

        Returns:
            dict: A dictionary where keys are user IDs and values are dictionaries containing various data paths.
        """
        # 排除 id 4（视频错误）, 2（时间太短）, 5 and 6 （时间戳对应不上）, 16 (weith problem)
        folders = [f for f in os.listdir(self.raw_data_path) if os.path.isdir(os.path.join(self.raw_data_path, f))]
        folders_without_problem = [f for f in folders if "problem" not in f and f not in ["2","4","5","6","16"]]

        dirs = {}
        for user_id in folders_without_problem:
                
            # 获取以 "DualCamera" 开头的路径
            Sensors_total_path = glob.glob(os.path.join(self.raw_data_path, user_id, "DualCamera*"))[0]

            # 获取以 "DualCamera" 开头的路径中包含的video_start_timestamp and video_end_time_stamp
            video_start_timestamp = split_string_number(Sensors_total_path.split('_')[1]) 
            video_end_timestamp = split_string_number(Sensors_total_path.split('_')[2]) 

            # 获取后置摄像头视频路径
            back_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "back_camera_17*"))[0]
            # 获取前置摄像头视频路径 
            front_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "front_camera_17*"))[0]

            # 获取后置摄像头数据路径
            back_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "back_camera_data*"))[0]
            # 获取前置摄像头数据路径
            front_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "front_camera_data*"))[0]

            # 获取 IMU 加速度数据路径
            imu_acc_data_path = glob.glob(os.path.join(Sensors_total_path, "imu_accelerometer_data*"))[0]
            # 获取 IMU 陀螺仪数据路径
            imu_gyro_data_path = glob.glob(os.path.join(Sensors_total_path, "imu_gyroscope_data*"))[0]

            # 获取以 "user_id_spO2_rr_data" 开头的路径
            label_total_path = glob.glob(os.path.join(self.raw_data_path, user_id, user_id + "_spO2_rr_data*"))[0]
            label_total_path = os.path.join(label_total_path, "v01")

            label_BVP_path = glob.glob(os.path.join(label_total_path, "BVP*"))[0]
            label_HR_path = glob.glob(os.path.join(label_total_path, "HR*"))[0]
            label_RR_path = glob.glob(os.path.join(label_total_path, "RR*"))[0]
            label_SpO2_path = glob.glob(os.path.join(label_total_path, "SpO2*"))[0]

            dirs[user_id]={
                         "back_camera_video_path":back_camera_video_path,
                         "front_camera_video_path":front_camera_video_path,
                         "back_camera_data_path":back_camera_data_path,
                         "front_camera_data_path":front_camera_data_path,
                         "imu_acc_data_path":imu_acc_data_path,
                         "imu_gyro_data_path":imu_gyro_data_path,
                         "video_start_timestamp":video_start_timestamp,
                         "video_end_timestamp":video_end_timestamp,
                         "label_BVP_path":label_BVP_path,
                         "label_HR_path":label_HR_path,
                         "label_RR_path":label_RR_path,
                         "label_SPO2_path":label_SpO2_path
                        }
        return dirs

    def load_video_data(self,user_id=1):
        """
        Loads and processes video data for a given user.

        Args:
            user_id (int): The user ID for which to load the video data. Default is 1.

        Returns:
            pd.DataFarme: Merged video frames from front and back cameras.
        """
        videos = {}
        if self.use_front:
            front_video = self.read_video_sync(self.dirs[f"{user_id}"]["front_camera_video_path"], visualize=False)
            if self.front_standard_type == "Standardized":
                front_video = self._standardized_frame(front_video)
            elif self.front_standard_type == "diff_normalize":
                front_video = self._diff_normalize_frame(front_video)
            videos['front_video'] = front_video

            # # 可视化检查，160张图片放在一页中
            # # n, h, w, c
            # video_length = front_video.shape[0]
            # fig, axes = plt.subplots(10, 16, figsize=(16, 10))
            # for i in range(10):
            #     for j in range(16):
            #         index = i * 16 + j
            #         if index < video_length:
            #             image = front_video[index]  # 假设数据集中每个样本的第一个元素是图像
            #             if isinstance(image, torch.Tensor):
            #                 image = image.numpy()  # 如果是Tensor，转换为numpy数组并调整维度
            #             if image.dtype == np.float32 or image.dtype == np.float64:
            #                 # Ensure values are in [0, 1]
            #                 image = np.clip(image, 0, 1)
            #             elif image.dtype == np.uint8:
            #                 # Ensure values are in [0, 255]
            #                 image = np.clip(image, 0, 255)
            #             print(image)
            #             axes[i, j].imshow(image)
            #             axes[i, j].axis('off')
            #         else:
            #             axes[i, j].axis('off')
            # plt.savefig(f"/data01/zt/Dataset_pt/anzhen_front_plot/dataset_images.png")


        if self.use_back:
            back_video = self.read_video_sync(self.dirs[f"{user_id}"]["back_camera_video_path"])
            if self.back_standard_type == "Standardized":
                back_video = self._standardized_frame(back_video)
            elif self.front_standard_type == "diff_normalize":
                back_video = self._diff_normalize_frame(back_video)                
            videos['back_video'] = back_video

        if len(videos) == 1:
            return self._handle_single_video(videos, user_id)
        else:
            return self.merge_video_frame(**videos, 
                                       front_video_timestamp_path=self.dirs[f"{user_id}"]["front_camera_data_path"],
                  
                                       back_video_timestamp_path=self.dirs[f"{user_id}"]["back_camera_data_path"])
    
    def _handle_single_video(self, videos, user_id):
        """处理单个视频模态的情况"""
        video_type = 'front' if 'front_video' in videos else 'back'
        timestamp_path = self.dirs[f"{user_id}"][f"{video_type}_camera_data_path"]
        
        # 创建单模态DataFrame
        timestamps = self.load_video_timestamp(timestamp_path)['timestamp']
        video_data = videos[video_type+"_video"]
        
        # 确保长度一致
        min_length = min(len(timestamps), len(video_data))
        return pd.DataFrame({
            'timestamp': timestamps[:min_length],
            f'{video_type}_video': [frame for frame in video_data[:min_length]]
        })

    def _process_frames_async(self):
        """
        Process frames in a background thread using face detection results from MediaPipe
        """
        box = box_ = None
        last_box_ = None
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            frame, process_flag = item
            
            # Step 1：Preprocess frame to handle different aspect ratios
            h, w, c = frame.shape
            if w > h:
                frame_ = frame[:, round((w - h) / 2):round((w - h) / 2) + h]
            else:
                frame_ = frame
                w = h
            
            # Step 2：Perform face detection using MediaPipe
            results = self.mp_face_mesh.process(frame_)
            if process_flag:
                if results.multi_face_landmarks:
                    landmark = np.array([(p.x * h / w + round((w - h)/2)/w, p.y) 
                                    for p in results.multi_face_landmarks[0].landmark])
                    
                    shape = alphashape.alphashape(landmark, 0)
                    if box is None:
                        box = np.array(shape.bounds).reshape(2, 2)
                    else:
                        w = 1/(1 + np.exp(-20*np.linalg.norm(np.array(shape.bounds).reshape(2, 2)-box)/np.multiply(*np.abs(box[0]-box[1]))))*2-1
                        box = np.array(shape.bounds).reshape(2, 2)*w+box*(1-w)
                    if box_ is None:
                        box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                    elif np.linalg.norm(np.round(box*frame.shape[1::-1]).astype(int).T - box_) > frame.size/10**5:
                        box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)

                    if box_ is not None:
                        result = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])], 
                                    (self.target_weight, self.target_height), 
                                    interpolation=cv2.INTER_AREA)
                    last_box_ = box_
            else:
                if last_box_ is not None:
                    result = cv2.resize(frame[slice(*last_box_[1]), slice(*last_box_[0])], 
                                                (self.target_weight, self.target_height), 
                                                interpolation=cv2.INTER_AREA)

            self.result_queue.put(result)

    def _process_frames_sync(self,frame,process_flag,box=None,box_=None):
        """
        Process frames in a background thread using face detection results from MediaPipe
        """
        
        # Step 1：Preprocess frame to handle different aspect ratios
        h, w, c = frame.shape
        if w > h:
            frame_ = frame[:, round((w - h) / 2):round((w - h) / 2) + h]
        else:
            frame_ = frame
            w = h
        
        # Step 2：Perform face detection using MediaPipe
        results = self.mp_face_mesh.process(frame_)
        if process_flag:
            if results.multi_face_landmarks:
                landmark = np.array([(p.x * h / w + round((w - h)/2)/w, p.y) 
                                for p in results.multi_face_landmarks[0].landmark])
                
                shape = alphashape.alphashape(landmark, 0)
                if box is None:
                    box = np.array(shape.bounds).reshape(2, 2)
                else:
                    w = 1/(1 + np.exp(-20*np.linalg.norm(np.array(shape.bounds).reshape(2, 2)-box)/np.multiply(*np.abs(box[0]-box[1]))))*2-1
                    box = np.array(shape.bounds).reshape(2, 2)*w+box*(1-w)
                if box_ is None:
                    box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)
                elif np.linalg.norm(np.round(box*frame.shape[1::-1]).astype(int).T - box_) > frame.size/10**5:
                    box_ = np.clip(np.round(box*frame.shape[1::-1]).astype(int).T, a_min=0, a_max=None)

                if box_ is not None:
                    result = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])], 
                                (self.target_weight, self.target_height), 
                                interpolation=cv2.INTER_AREA)
            else:
                result = cv2.resize(frame_, (self.target_weight, self.target_height), interpolation=cv2.INTER_AREA)
                return result,box,box_
        else:
            if box_ is not None:
                result = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])], 
                                            (self.target_weight, self.target_height), 
                                            interpolation=cv2.INTER_AREA)
            else:
                result = cv2.resize(frame_, (self.target_weight, self.target_height), interpolation=cv2.INTER_AREA)
                return result,box,box_
        
        return result,box,box_

    def read_video_async(self, video_path, chunk_size=100):
        """
        Reads video frames from a video file in chunks to avoid memory issues.

        Args:
            video_path (str): Path to the video file.
            chunk_size (int): Number of frames to process at a time. Default is 100.

        Returns:
            numpy.ndarray: A numpy array containing all video frames with shape (T, H, W, 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_type = "front" if "front" in video_path else "back"

        print(f"Video width: {width}, Video height: {height}, Video FPS: {fps:.2f}, Video length: {length}")

        # 启动front处理线程
        if video_type == "front":
            # clear the queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            worker = threading.Thread(target=self._process_frames)
            worker.daemon = True
            worker.start()

        all_chunks = []
        with tqdm(total=length//chunk_size, desc="Reading video frames") as pbar:

            frames = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    print(f"Warning: Empty frame encountered in video {video_path}")
                    continue

                # 分为front和back处理
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if video_type == "front":
                    process_flag = (frame_count % self.omitted_frames == 0)
                    self.frame_queue.put((frame, process_flag))
                    frame_count += 1

                    # 异步获取结果
                    while not self.result_queue.empty():
                        resized_frame = self.result_queue.get()
                        if resized_frame is not None:
                            frames.append(resized_frame)
                            if len(frames) == chunk_size:
                                all_chunks.append(np.array(frames))
                                frames = []
                                pbar.update(1)

                elif video_type == "back":
                    resized_frame = cv2.resize(frame,(self.target_weight, self.target_height), interpolation=cv2.INTER_AREA)
                    frames.append(resized_frame)
                    if len(frames) == chunk_size:
                        all_chunks.append(np.array(frames))
                        frames = []
                        pbar.update(1)

            # 处理剩余的帧
            if frames:
                all_chunks.append(np.array(frames))

        cap.release()
        self.frame_queue.put(None) # 退出front_process线程
        all_frames = np.concatenate(all_chunks, axis=0)
        print(f"Total frames read: {all_frames.shape[0]}")
        return all_frames
    
    def read_video_sync(self,video_path,video_type="Front", chunk_size=100, visualize=False):
        """
        Reads video frames from a video file in chunks to avoid memory issues.

        Args:
            video_path (str): Path to the video file.
            chunk_size (int): Number of frames to process at a time. Default is 100.
            visualize (bool, optional): If True, displays the first frame of the video. Default is False.

        Returns:
            numpy.ndarray: A numpy array containing all video frames with shape (T, H, W, 3).
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # video_type = "front" if "front" in video_path else "back"

        print(f"Video width: {width}, Video height: {height}, Video FPS: {fps:.2f}, Video length: {length}")

        all_chunks = []
        with tqdm(total=int(length/chunk_size), desc="Reading video frames",file=sys.stdout) as pbar:

            frames = []
            frame_count = 0
            box = box_ = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    print(f"Warning: Empty frame encountered in video {video_path}")
                    continue

                # 分为front和back处理
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

                if video_type == "Front":
                    flag = (frame_count % self.omitted_frames == 0)
                    resized_frame,box,box_ = self._process_frames_sync(frame,flag,box,box_)

                    frames.append(resized_frame)
                    frame_count += 1
                    if len(frames) == chunk_size:
                        all_chunks.append(np.array(frames))
                        frames = []
                        pbar.update(1)

                elif video_type == "Back":
                    resized_frame = cv2.resize(frame,(self.target_weight, self.target_height), interpolation=cv2.INTER_AREA)
                    frames.append(resized_frame)
                    if len(frames) == chunk_size:
                        all_chunks.append(np.array(frames))
                        frames = []
                        pbar.update(1)
                
                if visualize:
                    plt.imshow(resized_frame)
                    plt.savefig("/data01/zt/Dataset_pt/anzhen_front_plot/test.jpg")

            # 处理剩余的帧
            if frames:
                all_chunks.append(np.array(frames))

        cap.release()
        all_frames = np.concatenate(all_chunks, axis=0)
        print(f"Total frames read: {all_frames.shape[0]}")

        return all_frames
    
    def _standardized_frame(self,frames):
        """
        Applies Z-score standardization to the frames.

        Args:
            frames (numpy.ndarray): Input sequence of frames with shape (num_frames, height, width, channels).

        Returns:
            numpy.ndarray: Resized and standardized sequence of frames with shape (num_frames, target_height, target_width, channels).
        """
        frames = frames.astype(np.float32)
        mean = np.mean(frames)
        std = np.std(frames)
        frames[np.isnan(frames)] = 0
        frames = (frames - mean) / std

        return frames

    def _diff_normalize_frame(self,frames):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = frames.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (frames[j + 1, :, :, :] - frames[j, :, :, :]) / (
                    frames[j + 1, :, :, :] + frames[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data
    
    def _standardized_imu(self, imu):

        """
        Applies Z-score standardization to the IMU data while preserving the timestamp column.

        Args:
            imu (torch.Tensor): Input IMU data with shape [N, D], where N is the number of samples
                                and D is the number of features (including timestamp as first column).

        Returns:
            torch.Tensor: Standardized IMU data with the same shape as input, where all features
                          except the first column (timestamp) are standardized.
        """
        
        # Apply Z-score standardization
        standardized_imu = (imu - imu.mean()) / imu.std()
        
        # Handle NaN values
        standardized_imu[torch.isnan(standardized_imu)] = 0
        
        return standardized_imu
    
    def _standardized_label(self, label):
        """
        Applies Z-score standardization to the label data.

        Args:
            label (pd.DataFrame): Input label data with 'timestamp' and other columns.

        Returns:
            pd.DataFrame: DataFrame with standardized values, preserving the timestamp column.
        """
        # Preserve timestamp column
        timestamp_col = label['timestamp']
        numeric_df = label.drop('timestamp', axis=1)
        
        # Apply Z-score standardization
        standardized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
        
        # Handle NaN values
        standardized_df = standardized_df.fillna(0)
        standardized_df[np.isnan(standardized_df)] = 0
        
        # Restore timestamp column
        standardized_df['timestamp'] = timestamp_col
        
        return standardized_df

    def _diff_normalize_label(self, label):
        """
        Calculate discrete difference in labels along the time-axis and normalize by its standard deviation.
        
        Args:
            label (pd.DataFrame): Input label data with 'timestamp' and other columns.
            
        Returns:
            pd.DataFrame: DataFrame with diff-normalized values, preserving the timestamp column.
        """
        # Drop timestamp column temporarily
        timestamp_col = label['timestamp']
        numeric_df = label.drop('timestamp', axis=1)
        
        # Calculate differences and normalize
        diff_df = numeric_df.diff()
        normalized_df = diff_df / diff_df.std()
        
        # Handle edge cases
        normalized_df = normalized_df.fillna(0)
        normalized_df[np.isnan(normalized_df)] = 0
        
        # Restore timestamp column
        normalized_df['timestamp'] = timestamp_col
        
        return normalized_df

    def merge_video_frame(self, front_video, back_video, front_video_timestamp_path, back_video_timestamp_path):
        """
        Merges front and back video frames based on their timestamps.

        Args:
            front_video (numpy.ndarray): Front video frames with shape (T, H, W, 3).
            back_video (numpy.ndarray): Back video frames with shape (T, H, W, 3).
            front_video_timestamp_path (str): Path to the timestamp file for the front video.
            back_video_timestamp_path (str): Path to the timestamp file for the back video.

        Returns:
            pd.DataFrame: A DataFrame containing the merged video frames aligned by timestamps.
        """

        # Step 2: Load the timestamp arrays for both front and back videos
        front_video_timestamp = self.load_video_timestamp(front_video_timestamp_path)
        back_video_timestamp = self.load_video_timestamp(back_video_timestamp_path)

        # Step 3: Create DataFrames to store video frames along with their timestamps
        # 处理前视频数据
        front_timestamp = front_video_timestamp['timestamp']
        front_timestamp = front_video_timestamp.iloc[1:]['frame'] # for SUMS dataset
        # 找出较短的长度
        front_min_length = min(front_video.shape[0], front_video_timestamp.shape[0])
        # 裁剪序列
        front_timestamp = front_timestamp[:front_min_length]
        front_video = front_video[:front_min_length]

        # 构建前视频的 DataFrame
        front_video_total = pd.DataFrame({
            'timestamp': front_timestamp,
            'front_video': [frame for frame in front_video]
        })
        # 处理后视频数据
        back_timestamp = back_video_timestamp['timestamp']
        # 找出较短的长度
        back_min_length = min(back_video.shape[0], back_video_timestamp.shape[0])
        # 裁剪序列
        back_timestamp = back_timestamp[:back_min_length]
        back_video = back_video[:back_min_length]

        # 构建后视频的 DataFrame
        back_video_total = pd.DataFrame({
            'timestamp': back_timestamp,
            'back_video': [frame for frame in back_video]
        })

        # Step 4: Merge the DataFrames based on timestamps using nearest neighbor alignment
        aligned_data = pd.merge_asof(front_video_total, back_video_total, on='timestamp', direction='nearest', suffixes=('_front', '_back'))
        
        front_video = None
        back_video = None
        front_video_total = None
        back_video_total = None

        return aligned_data

    def load_video_timestamp(self,video_timestamp_path):
        """
        Loads video timestamps from a CSV file.

        Args:
            video_timestamp_path (str): Path to the timestamp file.

        Returns:
            pd.DataFrame: A DataFrame containing the timestamps and corresponding frame numbers.
        """
        df = pd.read_csv(video_timestamp_path, header=None, names=['timestamp', 'frame'])
        # print(f"video_timestamp_length:{df.iloc[-1,:]}")
        df = df.iloc[:-1,:]
        return df

    def load_imu_data(self,user_id=1):
        """
        Loads and merges IMU data for a given user.

        Args:
            user_id (int): The user ID for which to load the IMU data. Default is 1.

        Returns:
            pd.DataFrame: The merged and resampled IMU data with shape (L, 1+6).
        """
        imu_acc_data_path = self.dirs[f"{user_id}"]["imu_acc_data_path"]
        imu_gyro_data_path = self.dirs[f"{user_id}"]["imu_gyro_data_path"]

        imu_acc_data = self.read_imu_acc_data(imu_acc_data_path)

        imu_gyro_data = self.read_imu_gyro_data(imu_gyro_data_path)

        merge_imu_data = self.merge_resample_imu_data(imu_acc_data,imu_gyro_data,self.target_Hz)

        return merge_imu_data

    def read_imu_acc_data(self,imu_acc_data_path):
        """
        Reads IMU accelerometer data from a CSV file.

        Args:
            imu_acc_data_path (str): Path to the IMU accelerometer data file.

        Returns:
            pd.DataFrame: A DataFrame containing the normalized IMU accelerometer data.
        """
        imu_acc_data = pd.read_csv(imu_acc_data_path)
        mean = imu_acc_data.iloc[:,1:].mean()
        std = imu_acc_data.iloc[:,1:].std()
        std[std == 0] = 1
        normalized_data = (imu_acc_data.iloc[:, 1:] - mean) / std
        imu_acc_data.iloc[:, 1:] = normalized_data
        # print(f"imu_acc_data_head:{imu_acc_data.head()}")

        return imu_acc_data
    
    def read_imu_gyro_data(self,imu_gyro_data_path):
        """
        Reads IMU gyroscope data from a CSV file.

        Args:
            imu_gyro_data_path (str): Path to the IMU gyroscope data file.

        Returns:
            pd.DataFrame: A DataFrame containing the normalized IMU gyroscope data.
        """
        imu_gyro_data = pd.read_csv(imu_gyro_data_path)
        mean = imu_gyro_data.iloc[:,1:].mean()
        std = imu_gyro_data.iloc[:,1:].std()
        std[std == 0] = 1
        normalized_data = (imu_gyro_data.iloc[:, 1:] - mean) / std
        imu_gyro_data.iloc[:, 1:] = normalized_data

        return imu_gyro_data
    
    def _process_imu_acc_data(self,imu_acc_data):
        """
        Processes IMU accelerometer data.

        Args:
            imu_acc_data (pd.DataFrame): IMU accelerometer data.

        Returns:
            pd.DataFrame: Processed IMU accelerometer data.
        """

        start_time = time.time()

        # 计算采样率
        fs = 30 

        imu_processed_data = []
        for i in range(imu_acc_data.shape[0]):

            imu_acc_smaple = imu_acc_data[i]
            
            # 获取整个信号
            raw_signal_x = imu_acc_smaple[:,0]
            raw_signal_y = imu_acc_smaple[:,1]
            raw_signal_z = imu_acc_smaple[:,2]

            # 信号处理
            baseline_removed_x, Ceemdan_combined_x, processed_x = ImuAccProcessor.process_signal(raw_signal_x, fs)
            baseline_removed_y, Ceemdan_combined_y, processed_y = ImuAccProcessor.process_signal(raw_signal_y, fs)
            baseline_removed_z, Ceemdan_combined_z, processed_z = ImuAccProcessor.process_signal(raw_signal_z, fs)

            # 更新数据
            imu_acc_smaple[:,0] = torch.tensor(processed_x)
            imu_acc_smaple[:,1] = torch.tensor(processed_y)
            imu_acc_smaple[:,2] = torch.tensor(processed_z)

            imu_processed_data.append(imu_acc_smaple)

        imu_processed_data = torch.stack(imu_processed_data, dim=0)
        end_time = time.time()  # End time measurement
        print(f"Runtime of _process_imu_acc_data: {end_time - start_time:.4f} seconds")

        return imu_processed_data

    def merge_resample_imu_data(self, imu_acc_data, imu_gyro_data, target_Hz, target_length=None):
        """
        Resamples IMU data to a target sampling frequency.

        Args:
            imu_acc_data (pd.DataFrame): IMU accelerometer data.
            imu_gyro_data (pd.DataFrame): IMU gyroscope data.
            target_Hz (int): Target sampling frequency in Hz.
            target_length (int, optional): Target length of the resampled data. Default is None.

        Returns:
            pd.DataFrame: The resampled IMU data.
        """
        # Merge accelerometer and gyroscope data based on timestamp
        aligned_data = pd.merge_asof(imu_acc_data, imu_gyro_data, on='timestamp', direction='nearest')
        aligned_data['timestamp'] = pd.to_datetime(aligned_data['timestamp'])
        # print(f"Aligned data shape: {aligned_data.shape}")
        
        resampled_data = self.resample_data(aligned_data, target_Hz, target_length)

        # print(f"Resampled data shape: {resampled_data.shape}")

        return resampled_data

    def merge_video_imu_data(self,user_id=1):
        """
        Merges video and IMU data for a given user.

        Args:
            user_id (int): The user ID for which to merge the data. Default is 1.

        Returns:
            pd.DataFrame: A DataFrame containing the merged video and IMU data.
        """
        video_data = self.load_video_data(user_id)
        imu_data = self.load_imu_data(user_id)

        if self.dataset_name == "tjk_multimodal":
            # TODO video长度需要处理！！！！
            # Merge video and imu data based on timestamp
            start_time = pd.to_datetime(self.dirs[f"{user_id}"]["video_start_timestamp"],unit='s')
            end_time = pd.to_datetime(self.dirs[f"{user_id}"]["video_end_timestamp"],unit='s')
            
            video_data["timestamp"] = pd.date_range(start=start_time, end=end_time, periods=len(video_data))
            
            # TODO imu需要处理！！！！
            imu_data["timestamp"] = pd.date_range(start=start_time, end=end_time, periods=len(imu_data))
            
            video_imu_data = pd.merge_asof(video_data, imu_data, on='timestamp', direction='nearest')
            print(f"video_imu_data shape: {video_imu_data.shape}")
            return video_imu_data
        
        elif self.dataset_name == "anzhen_multimodal":
            # video and imu have the same timestamp
            start_time = pd.to_datetime(self.dirs[f"{user_id}"]["video_start_timestamp"],unit='s')
            end_time = pd.to_datetime(self.dirs[f"{user_id}"]["video_end_timestamp"],unit='s')
            
            video_data["timestamp"] = pd.to_datetime(video_data["timestamp"])
            imu_data["timestamp"] = pd.to_datetime(imu_data["timestamp"])
            
            video_imu_data = pd.merge_asof(video_data, imu_data, on='timestamp', direction='nearest')
            video_imu_data["timestamp"] = pd.date_range(start=start_time, end=end_time, periods=len(video_data))
            print(f"video_imu_data shape: {video_imu_data.shape}")
            return video_imu_data

    def load_label_data(self,user_id=1):
        """
        Loads label data for a given user.

        Args:
            user_id (int): The user ID for which to load the label data. Default is 1.

        Returns:
            pd.DataFrame: The loaded label data.
        """        
        loaded_label_data = {}
        for label in self.labels:
            label_path = self.dirs[f"{user_id}"][f"label_{label.upper()}_path"]
            if self.label_standard_type == "Standardized":
                temp_data = pd.read_csv(label_path)
                loaded_label_data[label] = self._standardized_label(temp_data)
            elif self.label_standard_type == "diff_normalize":
                temp_data = pd.read_csv(label_path)
                loaded_label_data[label] = self._diff_normalize_label(temp_data)
            else:
                loaded_label_data[label] = pd.read_csv(label_path)

        label = self.merge_labels_data(Phys_indicators=loaded_label_data)
        loaded_label_data = None
        return label

    def resample_data(self, raw_data, target_Hz, target_length=None):
        """
        Resamples raw data to a target sampling frequency.

        Args:
            raw_data (pd.DataFrame): The raw data to be resampled.
            target_Hz (int): The target sampling frequency in Hz.
            target_length (int, optional): The target length of the resampled data. Default is None.

        Returns:
            pd.DataFrame: The resampled data.
        """
        # Convert the 'timestamp' column to datetime format
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'],unit='s')

        # Get the start and end times of the raw data
        start_time = raw_data['timestamp'].min()
        end_time = raw_data['timestamp'].max()

        # Generate new timestamps based on the target sampling frequency or target length
        if target_length is not None:
            # If target_length is specified, generate timestamps with the specified number of periods
            new_timestamps = pd.date_range(start=start_time, end=end_time, periods=target_length)
        else:
            # If target_length is not specified, generate timestamps based on the target sampling frequency
            new_timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{1000 // target_Hz}ms')

        # Create a DataFrame with the new timestamps
        new_timestamps_df = pd.DataFrame(new_timestamps, columns=['timestamp'])

        # Merge the raw data with the new timestamps using pd.merge_asof
        # This will resample the raw data to the new timestamps by finding the nearest match
        resampled_data = pd.merge_asof(new_timestamps_df, raw_data, on='timestamp', direction='nearest')

        return resampled_data

    def merge_labels_data(self,Phys_indicators,target_Hz=30, target_length=None):
        """
        Resamples and merges physiological indicator data.

        Args:
            Phys_indicators (dict): A dictionary of DataFrames containing physiological indicator data.
            target_Hz (int): Target sampling frequency. Default is 30 Hz.
            target_length (int, optional): Target length of the resampled data. Default is None.

        Returns:
            pd.DataFrame: The merged and resampled physiological indicator data.
        """
        # Step 1: Resample each Physiological indicator data
        resampled_data_list = []
        for label, data in Phys_indicators.items():
            resampled_data = self.resample_data(data, target_Hz, target_length)
            resampled_data_list.append(resampled_data)
        
        # Step 2: Merge all resampled Physiological indicators data based on timestamp
        if len(resampled_data_list) == 1:
            aligned_data = resampled_data_list[0]
        else:
            aligned_data = resampled_data_list[0]
            for data in resampled_data_list[1:]:
                aligned_data = pd.merge_asof(aligned_data, data, on='timestamp', direction='nearest')

        # Ensure the timestamp column is in datetime format
        aligned_data['timestamp'] = pd.to_datetime(aligned_data['timestamp'])
        # print(f"Aligned data shape: {aligned_data.shape}")

        return aligned_data

    def load_merged_all_data(self,user_id=1):
        """
        Loads and merges all data (video, IMU, and labels) for a given user.

        Args:
            user_id (int): The user ID for which to load the data. Default is 1.

        Returns:
            pd.DataFrame: The merged data containing the video, IMU, and label data of target user.
            Keys: ['timestamp', 'front_video', 'back_video', 'accel_x', 'accel_y','accel_z', 'gyro_x',
              'gyro_y', 'gyro_z', 'bvp', 'hr', 'rr', 'spo2']
        """
        # Step 1: Load video data and label data
        print(f"\033[91mUser_{user_id}_data_loading_start\033[0m")
        video_imu_data = self.merge_video_imu_data(user_id)
        label_data = self.load_label_data(user_id)

        # Step 2: Merger features and label 
        if self.dataset_name == "tjk_multimodal":
            # TODO label和data之间的偏移,（都在PC）
            all_data = pd.merge_asof(video_imu_data, label_data, on='timestamp', direction='nearest')
        elif self.dataset_name == "anzhen_multimodal":
            # TODO 假设开头对齐，取前面一段数据
            # calculate the diff timestamp
            timestamp_gap = label_data['timestamp'].iloc[0] - video_imu_data['timestamp'].iloc[0]
            # add the gap to the video_imu_data
            video_imu_data['timestamp'] = video_imu_data['timestamp'] + timestamp_gap
            all_data = pd.merge_asof(video_imu_data, label_data, on='timestamp', direction='nearest')

        video_imu_data = None
        label_data = None
        del video_imu_data
        del label_data

        print(f"\033[91mUser_{user_id}_data_loading_done\033[0m")

        return all_data

    def _create_loso_data_loader(self, call_count, test_user_number):
        """
        Creates a data loader for the target users, where the test data is from one user and the training data is from all other users.

        Args:
            call_count (int): The index of the user to be used as the test user.
            test_user_number (int): The number of users to be used as test users.

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        if test_user_number > 1:
            test_user_id = np.random.choice(list(self.dirs.keys()),test_user_number)
            print(f"test_user_id: {test_user_id}")
            train_user_ids = [user_id for user_id in self.dirs.keys() if user_id not in test_user_id]
            data_train = pd.concat([self.load_merged_all_data(user_id) for user_id in train_user_ids])
            data_test = pd.concat([self.load_merged_all_data(test_user_id) for test_user_id in test_user_id])

        else:
            test_user_id = list(self.dirs.keys())[call_count]
            print(f"test_user_id: {test_user_id}")
            train_user_ids = [user_id for user_id in self.dirs.keys() if user_id != test_user_id]
            data_train = pd.concat([self.load_merged_all_data(user_id) for user_id in train_user_ids])
            data_test = self.load_merged_all_data(test_user_id)

        
        train_dataset = self._create_dataset(data_train)
        test_dataset = self._create_dataset(data_test)
        data_train = None
        data_test = None

        # 保存 Dataset 对象
        train_dataset.save_dataset_to_hdf("/data02/tjk/zt/Dataset_pt/train_dataset.h5")
        test_dataset.save_dataset_to_hdf("/data02/tjk/zt/Dataset_pt/test_dataset.h5")

    def _apply_processing(self, data, processing_type, processing_functions):
        if processing_type in processing_functions:
            processed_data = []
            for sample in data:
                processed_sample = processing_functions[processing_type](sample)
                processed_data.append(processed_sample)
            return np.stack(processed_data)
        return data

    def _create_dataset(self, data):
        """
        Creates a dataset from the provided data by converting it into tensors.

        Args:
            data (pd.DataFrame): The input data containing features and labels.

        Returns:
            MultimodalDataset: A dataset object containing the features and labels.
        """
        features = {
            "imu_acc": torch.tensor(data[["accel_x", "accel_y", "accel_z"]].values, dtype=torch.float32),
            "imu_gyro": torch.tensor(data[["gyro_x", "gyro_y", "gyro_z"]].values, dtype=torch.float32)
        }

        processing_functions = {
            "Standardized": self._standardized_frame,
            "diff_normalize": self._diff_normalize_frame
        }
        
        # video 切片后标准化
        if self.use_front:
            features["video_front"] = torch.tensor(np.stack(data["front_video"]), dtype=torch.float32)
            features["video_front"] = self._split_into_sequences(features["video_front"])
            features["video_front"] = self._apply_processing(features["video_front"], self.front_standard_type, processing_functions)

        if self.use_back:
            features["video_back"] = torch.tensor(np.stack(data["back_video"]), dtype=torch.float32)
            features["video_back"] = self._split_into_sequences(features["video_back"])           
            features["video_back"] = self._apply_processing(features["video_back"], self.back_standard_type, processing_functions)

        # imu 切片后进行前处理
        features["imu_acc"] = self._split_into_sequences(features["imu_acc"])
        features["imu_gyro"] = self._split_into_sequences(features["imu_gyro"]) 
        # features["imu_acc"] = self._process_imu_acc_data(features["imu_acc"]) # 对acc进行滤波等前处理

        if self.imu_standard_type in processing_functions:
            features["imu_acc"] = self._apply_processing(features["imu_acc"], self.imu_standard_type, {"Standardized":self._standardized_imu})
            features["imu_gyro"] = self._apply_processing(features["imu_gyro"], self.imu_standard_type, {"Standardized":self._standardized_imu})

        # label 切片
        if self.dataset_name == "tjk_multimodal":
            labels = self._split_into_sequences(torch.tensor(data[["bvp", "hr", "rr", "spo2"]].values, dtype=torch.float32))
            return MultimodalDataset(
                features=features,
                labels={
                    'bvp': labels[:,:,0],
                    'hr': labels[:,:,1],
                    'rr': labels[:,:,2],
                    'spo2': labels[:,:,3]
                },
                use_front=self.use_front,
                use_back=self.use_back
            )

        elif self.dataset_name == "anzhen_multimodal":
            labels = self._split_into_sequences(torch.tensor(data[["bvp", "hr", "spo2"]].values, dtype=torch.float32))
            return MultimodalDataset(
                features=features,
                labels={
                    'bvp': labels[:,:,0],
                    'hr': labels[:,:,1],
                    'spo2': labels[:,:,2]
                },
                use_front=self.use_front,
                use_back=self.use_back
            )        

    def _split_into_sequences(self,data):
        """
        Splits the input data into sequences of a fixed length.

        Args:
            data (torch.Tensor): The input data to be split into sequences.

        Returns:
            torch.Tensor: A tensor containing the sequences of data.
        """
        num_sequences = data.shape[0] // self.seq_len
        sequences = [data[i * self.seq_len:(i + 1) * self.seq_len] for i in range(num_sequences)]
        return torch.stack(sequences) # [num_sequences, seq_len, num_features]
    
    def _test_user_video_timestamp_length(self, user_id = 1):
        
        print("Video data:")
        self.read_video_sync(self.dirs[str(user_id)]["front_camera_video_path"])
        self.read_video_sync(self.dirs[str(user_id)]["back_camera_video_path"])

        print("TimeStamp data:")
        front_video_timestamp = self.load_video_timestamp(self.dirs[str(user_id)]["front_camera_data_path"])
        back_video_timestamp = self.load_video_timestamp(self.dirs[str(user_id)]["back_camera_data_path"])

    def _test_save_datasets(self):
        dir_path = r"/data01/zt/Dataset_pt/tjk_multimodal"
        user_ids = self.dirs.keys()
        for user_id in user_ids:
            gc.collect()  # 手动触发垃圾回收
            dataset_df = self.load_merged_all_data(user_id)
            dataset = self._create_dataset(dataset_df)
            torch.save(dataset,f"{dir_path}/front_{self.front_standard_type}_label_{self.label_standard_type}_dataset_{user_id}.pth",pickle_protocol=4)


class MultimodalDataLoaderAnZhen(MultimodalDataLoader):
    """The data loader for the AnZhen dataset."""

    def __init__(self, config):
        self.dataset_name = "anzhen_multimodal"
        self.raw_data_path = r"/data01/tjk/安贞医院合作"
        self.use_front = True
        self.use_back = True

        # standard config 
        self.front_standard_type = config.front_standard_type if self.use_front else None
        self.back_standard_type  = config.back_standard_type if self.use_back else None
        self.imu_standard_type = config.imu_standard_type
        self.label_standard_type = config.label_standard_type
        
        # resize config
        self.target_height = config.resize_height
        self.target_weight = config.resize_width
        self.target_Hz = config.target_hz
        self.seq_len = config.seq_len

        # test config
        self.test_size = config.test_size
        self.batch_size = config.batch_size
        self.dirs = self.get_user_data_dirs()
        print(len(self.dirs))
        # print(self.dirs.keys())
        self.labels = ["bvp","hr","spo2"]
        self.user_number = len(self.dirs.keys())
        self.user_ids = self.dirs.keys()

        # face detector config
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.omitted_frames = 1
        self.buffer_size = 100
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.result_queue = Queue(maxsize=self.buffer_size)

    def get_user_data_dirs(self):
        """
        Collects data directories and relevant file paths for each user.

        Returns:
            dict: A dictionary where keys are user IDs and values are dictionaries containing various data paths.
        """
        feature_folders = [f for f in os.listdir(self.raw_data_path+"/DualCamera")]
        label_folders = [f for f in os.listdir(self.raw_data_path+"/anzhenData")]
        
        # folders_without_problem = [f for f in folders if "problem" not in f and f not in ["2","4","5","6","16"]]

        dirs = {}
        for path in feature_folders:
                
            Sensors_total_path = os.path.join(self.raw_data_path, "DualCamera", path)

            # process the feature data
            try:
                user_informaion_path = glob.glob(os.path.join(Sensors_total_path, "nurse*"))[0]
                with open(user_informaion_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # 这里数据可以扩展
                    user_id = data["用户ID"]
                    DBP = data["舒张压"]
                    SBP = data["收缩压"]
                    print("User Information:", data)
            except:
                continue
                print("Miss nurse_evaluation.txt")

            # 获取以 "DualCamera" 开头的路径中包含的video_start_timestamp and video_end_time_stamp
            video_start_timestamp = split_string_number(Sensors_total_path.split('_')[1]) 
            video_end_timestamp = split_string_number(Sensors_total_path.split('_')[2]) 

            # 获取后置摄像头视频路径
            back_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "back_camera*"))[0]
            # 获取前置摄像头视频路径 
            front_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "front_camera*"))[0]

            # 获取后置摄像头数据路径
            back_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "back_camera_data*"))[0]
            # 获取前置摄像头数据路径
            front_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "front_camera_data*"))[0]

            # 获取 IMU 加速度数据路径
            imu_acc_data_path = glob.glob(os.path.join(Sensors_total_path, "imu_accelerometer_data*"))[0]
            # 获取 IMU 陀螺仪数据路径
            imu_gyro_data_path = glob.glob(os.path.join(Sensors_total_path, "imu_gyroscope_data*"))[0]
            
            if user_id in label_folders:

                Label_total_path = os.path.join(self.raw_data_path, "anzhenData")
                # 获取以 "user_id_spO2_rr_data" 开头的路径
                label_total_path = glob.glob(os.path.join(Label_total_path, user_id))[0]
                label_total_path = os.path.join(label_total_path, "v01")

                label_BVP_path = glob.glob(os.path.join(label_total_path, "BVP*"))[0]
                label_HR_path = glob.glob(os.path.join(label_total_path, "HR*"))[0]
                label_SpO2_path = glob.glob(os.path.join(label_total_path, "SpO2*"))[0]

            else:
                print("Error: User ID not found in label folders.")
                continue

            # process the label data
            dirs[user_id]={
                         "back_camera_video_path":back_camera_video_path,
                         "front_camera_video_path":front_camera_video_path,
                         "back_camera_data_path":back_camera_data_path,
                         "front_camera_data_path":front_camera_data_path,
                         "imu_acc_data_path":imu_acc_data_path,
                         "imu_gyro_data_path":imu_gyro_data_path,
                         "video_start_timestamp":video_start_timestamp,
                         "video_end_timestamp":video_end_timestamp,
                         "label_BVP_path":label_BVP_path,
                         "label_HR_path":label_HR_path,
                         "label_SPO2_path":label_SpO2_path,
                         "SBP":SBP,
                         "DBP":DBP
                        }
        return dirs


    def _test_save_datasets(self):
        dir_path = r"/data01/zt/Dataset_pt/anzhen_multimodal"
        user_ids = self.dirs.keys()
        for user_id in user_ids:
            gc.collect()  # 手动触发垃圾回收
            dataset_df = self.load_merged_all_data(user_id)
            dataset = self._create_dataset(dataset_df)
            torch.save(dataset,f"{dir_path}/front_{self.front_standard_type}_label_{self.label_standard_type}_dataset_{user_id}.pth")


class MultimodalDataLoaderSUMS(MultimodalDataLoader):

    def __init__(self, config):
        self.dataset_name = "SUMS_multimodal"
        self.raw_data_path = r"/data01/zt/Dataset_Raw/SUMS_Raw"

        # standard config 
        self.front_standard_type = config.front_standard_type 
        self.back_standard_type  = config.back_standard_type 
        self.label_standard_type = config.label_standard_type
        
        # resize config
        self.target_height = config.resize_height
        self.target_weight = config.resize_width
        self.target_Hz = 30 
        self.seq_len = config.seq_len

        # test config
        self.test_size = config.test_size
        self.batch_size = config.batch_size
        self.dirs = self.get_user_data_dirs()
        print(len(self.dirs))
        # print(self.dirs.keys())
        self.labels = ["bvp","hr","rr","spo2"]
        self.user_number = len(self.dirs.keys())
        self.user_ids = self.dirs.keys()

        # face detector config
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.omitted_frames = 1
        self.buffer_size = 100
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.result_queue = Queue(maxsize=self.buffer_size)

    def get_user_data_dirs(self):
        """
        Collects data directories and relevant file paths for each user.

        Returns:
            dict: A dictionary where keys are user IDs and values are dictionaries containing various data paths.
        """
        folders = [f for f in os.listdir(self.raw_data_path) if os.path.isdir(os.path.join(self.raw_data_path, f))]

        dirs = {}
        for user_id in folders:
            dirs[user_id] = {}
            for secnario in ["v01", "v02", "v03", "v04"]:
                # 获取以 "user_id_spO2_rr_data" 开头的路径
                back_camera_video_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"video_ZIP_H264_finger*"))[0]
                front_camera_video_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"video_ZIP_H264_face*"))[0]
                back_camera_data_path = front_camera_data_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"frames_timestamp*"))[0]
                label_BVP_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"BVP*"))[0]
                label_HR_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"HR*"))[0]
                label_RR_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"RR*"))[0]
                label_SpO2_path = glob.glob(os.path.join(self.raw_data_path, user_id, secnario,"SpO2*"))[0]                

                dirs[user_id][secnario]={
                        "back_camera_video_path":back_camera_video_path,
                        "front_camera_video_path":front_camera_video_path,
                        "back_camera_data_path":back_camera_data_path,
                        "front_camera_data_path":front_camera_data_path,
                        "label_BVP_path":label_BVP_path,
                        "label_HR_path":label_HR_path,
                        "label_RR_path":label_RR_path,
                        "label_SPO2_path":label_SpO2_path
                    }
        return dirs


    def load_video_data(self,user_id=1,secnario='v01'):
        """
        Loads and processes video data for a given user.

        Args:
            user_id (int): The user ID for which to load the video data. Default is 1.

        Returns:
            pd.DataFarme: Merged video frames from front and back cameras.
        """
        videos = {}

        front_video = self.read_video_sync(self.dirs[user_id][secnario]["front_camera_video_path"],video_type="Front")
        if self.front_standard_type == "Standardized":
            front_video = self._standardized_frame(front_video)
        elif self.front_standard_type == "diff_normalize":
            front_video = self._diff_normalize_frame(front_video)

        back_video = self.read_video_sync(self.dirs[user_id][secnario]["back_camera_video_path"],video_type="Back",)
        if self.back_standard_type == "Standardized":
            back_video = self._standardized_frame(back_video)
        elif self.front_standard_type == "diff_normalize":
            back_video = self._diff_normalize_frame(back_video)       
        
        videos['front_video'] = front_video
        videos['back_video'] = back_video

        return self.merge_video_frame(**videos, 
                                    front_video_timestamp_path=self.dirs[user_id][secnario]["front_camera_data_path"],
                                    back_video_timestamp_path=self.dirs[user_id][secnario]["back_camera_data_path"])


    def merge_video_frame(self, front_video, back_video, front_video_timestamp_path, back_video_timestamp_path):
        """
        Merges front and back video frames based on their timestamps.

        Args:
            front_video (numpy.ndarray): Front video frames with shape (T, H, W, 3).
            back_video (numpy.ndarray): Back video frames with shape (T, H, W, 3).
            front_video_timestamp_path (str): Path to the timestamp file for the front video.
            back_video_timestamp_path (str): Path to the timestamp file for the back video.

        Returns:
            pd.DataFrame: A DataFrame containing the merged video frames aligned by timestamps.
        """

        # Step 2: Load the timestamp arrays for both front and back videos
        front_video_timestamp = pd.read_csv(front_video_timestamp_path, header=0, names=['frame','timestamp'])
        back_video_timestamp = pd.read_csv(back_video_timestamp_path, header=0, names=['frame','timestamp'])

        # Step 3: Create DataFrames to store video frames along with their timestamps
        # 处理前视频数据
        front_timestamp = pd.to_datetime(front_video_timestamp['timestamp'],unit='s')
        print(front_timestamp.head)
        # 找出较短的长度
        front_min_length = min(front_video.shape[0], front_video_timestamp.shape[0])
        # 裁剪序列
        front_timestamp = front_timestamp[:front_min_length]
        front_video = front_video[:front_min_length]

        # 构建前视频的 DataFrame
        front_video_total = pd.DataFrame({
            'timestamp': front_timestamp,
            'front_video': [frame for frame in front_video]
        })

        # 处理后视频数据
        back_timestamp = pd.to_datetime(back_video_timestamp['timestamp'])
        # 找出较短的长度
        back_min_length = min(back_video.shape[0], back_video_timestamp.shape[0])
        # 裁剪序列
        back_timestamp = back_timestamp[:back_min_length]
        back_video = back_video[:back_min_length]

        # 构建后视频的 DataFrame
        back_video_total = pd.DataFrame({
            'timestamp': back_timestamp,
            'back_video': [frame for frame in back_video]
        })

        # Step 4: Merge the DataFrames based on timestamps using nearest neighbor alignment
        aligned_data = pd.merge_asof(front_video_total, back_video_total, on='timestamp', direction='nearest', suffixes=('_front', '_back'))
        
        aligned_data['timestamp'] = pd.to_datetime(aligned_data['timestamp'])
        front_video = None
        back_video = None
        front_video_total = None
        back_video_total = None

        return aligned_data


    def load_label_data(self,user_id=1,secnario="v01"):
        """
        Loads label data for a given user.

        Args:
            user_id (int): The user ID for which to load the label data. Default is 1.

        Returns:
            pd.DataFrame: The loaded label data.
        """        
        loaded_label_data = {}
        for label in self.labels:
            label_path = self.dirs[user_id][secnario][f"label_{label.upper()}_path"]
            if self.label_standard_type == "Standardized":
                temp_data = pd.read_csv(label_path)
                loaded_label_data[label] = self._standardized_label(temp_data)
            elif self.label_standard_type == "diff_normalize":
                temp_data = pd.read_csv(label_path)
                loaded_label_data[label] = self._diff_normalize_label(temp_data)
            else:
                loaded_label_data[label] = pd.read_csv(label_path)

        label = self.merge_labels_data(Phys_indicators=loaded_label_data,target_Hz=self.target_Hz)

        loaded_label_data = None
        return label


    def load_user_secnario_data(self,user_id=1,secnario='v01'):
        """
        Loads and merges all data (video, IMU, and labels) for a given user.

        Args:
            user_id (int): The user ID for which to load the data. Default is 1.

        Returns:
            pd.DataFrame: The merged data containing the video, IMU, and label data of target user.
            Keys: ['timestamp', 'front_video', 'back_video', 'bvp', 'hr', 'rr', 'spo2']
        """
        # Step 1: Load video data and label data
        print(f"\033[91mUser_{user_id}_{secnario}_data_loading_start\033[0m")

        video_data = self.load_video_data(user_id,secnario)
        label_data = self.load_label_data(user_id,secnario)

        # Step 2: Resample video data to the target Hz
        video_data = self.resample_data(video_data, target_Hz=self.target_Hz) 
        
        print(video_data.head)
        print(label_data.head)
        # Step 2: Merger features and label 
        all_data = pd.merge_asof(video_data, label_data, on='timestamp', direction='nearest')

        video_imu_data = None
        label_data = None
        del video_imu_data
        del label_data

        print(f"\033[91mUser_{user_id}_{secnario}_data_loading_done\033[0m")

        return all_data


    def load_user_data(self,user=1):

        for secnario in ["v01","v02","v03","v04"]:
            secnario_data = self.load_user_secnario_data(user_id=user,secnario=secnario)
            if secnario == "v01":
                all_data = secnario_data
            else:
                all_data = pd.concat([all_data,secnario_data],axis=0)
        
        return all_data


    def _create_dataset(self, data):
        """
        Creates a dataset from the provided data by converting it into tensors.

        Args:
            data (pd.DataFrame): The input data containing features and labels.

        Returns:
            MultimodalDataset: A dataset object containing the features and labels.
        """
        features = {}

        processing_functions = {
            "Standardized": self._standardized_frame,
            "diff_normalize": self._diff_normalize_frame
        }
        
        features["video_front"] = torch.tensor(np.stack(data["front_video"]), dtype=torch.float32)
        features["video_front"] = self._split_into_sequences(features["video_front"])
        features["video_front"] = self._apply_processing(features["video_front"], self.front_standard_type, processing_functions)

        features["video_back"] = torch.tensor(np.stack(data["back_video"]), dtype=torch.float32)
        features["video_back"] = self._split_into_sequences(features["video_back"])           
        features["video_back"] = self._apply_processing(features["video_back"], self.back_standard_type, processing_functions)

        labels = self._split_into_sequences(torch.tensor(data[["bvp", "hr", "rr", "spo2"]].values, dtype=torch.float32))
        
        return MultimodalDataset(
            features=features,
            labels={
                'bvp': labels[:,:,0],
                'hr': labels[:,:,1],
                'rr':labels[:,:,2],
                'spo2': labels[:,:,3]
            },
            use_front=True,
            use_back=True
        )   


    def _test_save_datasets(self):
        dir_path = r"/data01/zt/Dataset_pt/SUMS_multimodal"
        user_ids = self.dirs.keys()
        # for user_id in ["060200"]:
        for user_id in ["060200","060201","060202","060203","060204","060205","060206","060207","060208","060209"]:
            gc.collect()  # 手动触发垃圾回收
            dataset_df = self.load_user_data(user_id)
            dataset = self._create_dataset(dataset_df)
            torch.save(dataset,f"{dir_path}/front_{self.front_standard_type}_label_{self.label_standard_type}_dataset_{user_id}.pth",
            pickle_protocol=5)


def convert_dataset_to_iterable_format(
    original_dataset_path, 
    output_dir, 
    subject_id="subject_001",
    use_front=True,
    use_back=True
):
    """
    将原 Map-style Dataset 转换为 IterableDataset 所需的分文件格式
    
    Args:
        original_dataset_path: str 
            原数据集路径（.pth 文件）
        output_dir: str 
            输出目录（每个被试一个子文件夹）
        subject_id: str 
            被试标识符（如 "subject_001"）
        use_front: bool 
            是否保留前视视频
        use_back: bool 
            是否保留后视视频
    """
    # 加载原数据集
    saved_data = torch.load(original_dataset_path)
    original_data = saved_data.data  # 假设原数据集是 MultimodalDataset 实例
    dataset_len = len(original_data["imu_acc"])
    if subject_id == '9': # 102-123
        indices = [i for i in range(dataset_len) if i < 100 or i > 125]
    elif subject_id == '10':
        indices = [i for i in range(dataset_len) if i not in [105]]
    elif subject_id == '12':
        indices = [i for i in range(dataset_len) if i not in [105]]
    elif subject_id == '13':
        indices = [i for i in range(dataset_len) if i not in [124,125,126]]
    elif subject_id == '15':
        indices = [i for i in range(dataset_len) if i not in [99,100,101,102]]
    else:
        indices = [i for i in range(dataset_len)]

    indices = [i for i in indices if i not in {0, 1, 2, dataset_len - 1}]
    original_data = {
    k1: v1[indices] if k1 != "labels" else {k2: v2[indices] for k2, v2 in v1.items()}
    for k1, v1 in original_data.items()}

    # 创建被试文件夹
    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    # ----------------------------
    # 1. 提取并保存各模态数据
    # ----------------------------
    modalities = {}
    
    # IMU 数据（必选）
    for modality in ["imu_acc", "imu_gyro"]:
        modalities[modality] = original_data[modality]
        torch.save(
            original_data[modality], 
            os.path.join(subject_dir, f"{modality}.pt"),
            pickle_protocol=5
        )

    # 视频数据（可选）
    if use_front and "video_front" in original_data:
        modalities["video_front"] = original_data["video_front"]
        torch.save(
            original_data["video_front"],
            os.path.join(subject_dir, "video_front.pt"),
            pickle_protocol=5
        )

    if use_back and "video_back" in original_data:
        modalities["video_back"] = original_data["video_back"]
        torch.save(
            original_data["video_back"],
            os.path.join(subject_dir, "video_back.pt"),
            pickle_protocol=5
        )

    # ----------------------------
    # 2. 保存标签
    # ----------------------------
    labels = original_data["labels"]
    torch.save(labels, os.path.join(subject_dir, "labels.pt"), pickle_protocol=5)

    # ----------------------------
    # 3. 生成 metadata.pt
    # ----------------------------
    metadata = {
        "total_samples": len(labels["bvp"]),
        "modalities": list(modalities.keys()),
        "label_types": list(labels.keys())
    }
    torch.save(metadata, os.path.join(subject_dir, "metadata.pt"), pickle_protocol=5)

    print(f"Converted {subject_id} data saved to {subject_dir}")


if __name__ == "__main__":
    
    # TODO anzhen数据集label和front的时间戳无法对应！！例如id：0060316427，front(1699089000.302,1699089036.845),label(1732536856.635437，1732536886.586144)
    # TODO 目前处理方法，直接使用整段数据进行对齐

    args = config.get_config()
    data_loader = MultimodalDataLoaderSUMS(config=args)
    data_loader._test_save_datasets()

    # for path in os.listdir("/data01/zt/Dataset_pt/tjk_multimodal"):
    #     if path.endswith(".pth"):
    #         # 读取数据集
    #         user_id = str(path.split(".")[0])[-2:]
    #         if '_' not in user_id:
    #             convert_dataset_to_iterable_format(os.path.join("/data01/zt/Dataset_pt/tjk_multimodal", path),
    #                                    "/data01/zt/Dataset_pt/Iterable/tjk_multimodal",
    #                                     user_id)
    #         print(user_id)


    # # 创建数据集
    # dataset = MultimodalIterableDataset(["/data01/zt/Dataset_pt/Iterable/anzhen_multimodal/0053018026"])

    # # 创建 DataLoader
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     num_workers=4,
    #     prefetch_factor=2,
    # )

    # # 训练循环
    # for batch in dataloader:
    #     front = batch["modals"]["video_front"]  # shape: [32, ...]
    #     hr = batch["labels"]["bvp"]            # shape: [32]