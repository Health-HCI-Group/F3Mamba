import numpy as np 
import pandas as pd
import os
import sys
import cv2
import glob
import re
from tqdm import tqdm
import torch
from queue import Queue
import alphashape
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

def split_string_number(num_str):
    if len(num_str) <= 10:
        return float(num_str)
    integer_part = num_str[:10]
    decimal_part = num_str[10:]
    result = float(f"{integer_part}.{decimal_part}")
    return result

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, use_front=True, use_back=True):
        self.data = {
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
        for modality in ["video_front", "video_back"]:
            if modality in self.data:
                sample["modals"][modality] = self.data[modality][idx]
        # 支持所有可能的标签类型
        all_label_keys = ["bvp", "hr", "rr", "spo2", "bp_sys", "bp_dia", "hrv"]
        for modality in all_label_keys:
            if modality in self.data["labels"]:
                sample["labels"][modality] = self.data["labels"][modality][idx]
        return sample

class MultimodalDataLoader():
    def __init__(self, config):
        self.dataset_name = config.dataset_name if hasattr(config, 'dataset_name') else "Lab_multimodal"
        print(f"dataset_name:{self.dataset_name}")
        self.raw_data_path = config.data_path
        self.use_front = config.use_front if hasattr(config, 'use_front') else True
        self.use_back = config.use_back if hasattr(config, 'use_back') else True

        self.front_standard_type = config.front_standard_type if self.use_front else None
        self.back_standard_type  = config.back_standard_type if self.use_back else None
        self.label_standard_type = config.label_standard_type
        
        self.target_height = config.resize_height
        self.target_weight = config.resize_width
        self.target_Hz = config.target_hz
        self.seq_len = config.seq_len

        # 根据数据集名称选择不同的标签配置
        if self.dataset_name == "Lab_multimodal":
            self.labels = ["bvp", "hr", "rr", "spo2"]
        elif self.dataset_name == "AZDualCamera":
            self.labels = ["bvp", "hr", "rr", "spo2", "bp_sys", "bp_dia"]
        else:
            self.labels = ["bvp", "hr", "rr", "spo2"]

        self.dirs = self.get_user_data_dirs()
        self.user_number = len(self.dirs.keys())
        self.user_ids = self.dirs.keys()

        # MediaPipe FaceLandmarker (new Tasks API, mediapipe>=0.10.9)
        # Requires model file download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
        import os as _os
        _model_paths = [
            '/tmp/face_landmarker.task',
            _os.path.expanduser('~/.cache/mediapipe/face_landmarker.task'),
        ]
        _model_path = None
        for _p in _model_paths:
            if _os.path.exists(_p):
                _model_path = _p
                break
        self._mp_facemesh_available = _model_path is not None
        if self._mp_facemesh_available:
            self._mp_facemesh_options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_model_path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._mp_facemesh = None  # lazy init
        else:
            print("⚠️  FaceLandmarker model not found. Using center-crop fallback.")
            print("   Download: wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -O /tmp/face_landmarker.task")
            self._mp_facemesh_available = False

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
        if self.dataset_name == "Lab_multimodal":
            return self._get_lab_multimodal_dirs()
        elif self.dataset_name == "AZDualCamera":
            return self._get_az_dual_camera_dirs()
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    def _get_lab_multimodal_dirs(self):
        """获取 Lab_multimodal 数据集目录结构"""
        folders = [f for f in os.listdir(self.raw_data_path) if os.path.isdir(os.path.join(self.raw_data_path, f))]
        folders_without_problem = [f for f in folders if "problem" not in f and f not in ["2","4","5","6","16"]]

        dirs = {}
        for user_id in folders_without_problem:
            Sensors_total_path = glob.glob(os.path.join(self.raw_data_path, user_id, "DualCamera*"))[0]
            video_start_timestamp = split_string_number(Sensors_total_path.split('_')[1]) 
            video_end_timestamp = split_string_number(Sensors_total_path.split('_')[2]) 
            back_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "back_camera_17*"))[0]
            front_camera_video_path = glob.glob(os.path.join(Sensors_total_path, "front_camera_17*"))[0]
            back_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "back_camera_data*"))[0]
            front_camera_data_path = glob.glob(os.path.join(Sensors_total_path, "front_camera_data*"))[0]
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
                         "video_start_timestamp":video_start_timestamp,
                         "video_end_timestamp":video_end_timestamp,
                         "label_BVP_path":label_BVP_path,
                         "label_HR_path":label_HR_path,
                         "label_RR_path":label_RR_path,
                         "label_SPO2_path":label_SpO2_path
                        }
        return dirs

    def _get_az_dual_camera_dirs(self):
        """
        获取安贞医院双摄数据集 (AZDualCamera) 的目录结构。
        
        数据组织格式:
            <subject_id>/V001/
                Face.mp4       - 前置摄像头视频 (1920x1080, 30fps)
                Finger.mp4     - 后置摄像头视频 (1920x1080, 30fps)
                Timestamp.csv  - 视频帧时间戳 (wall_ms, sensor_ts_sec, relative_sec, frame_number)
                PPG_GT.csv     - 血氧仪PPG标签 (timestamp_ms, bvp)
                IMU.csv        - IMU传感器数据
                Metrics.txt    - 受试者元信息 (ID, BP, Age, Gender, Height, Weight)
                Analysis.txt   - 视频后处理分析报告
        """
        # 扫描所有受试者目录（排除以 _ 开头的测试/未知目录）
        all_items = os.listdir(self.raw_data_path)
        subject_dirs = []
        for item in all_items:
            item_path = os.path.join(self.raw_data_path, item)
            if os.path.isdir(item_path) and not item.startswith('_') and not item.startswith('.'):
                subject_dirs.append(item)
        
        # 也包含 _测试 等测试目录（如果配置允许）
        if hasattr(self, 'include_test_subjects') and self.include_test_subjects:
            for item in all_items:
                item_path = os.path.join(self.raw_data_path, item)
                if os.path.isdir(item_path) and item.startswith('_') and not item.startswith('.'):
                    subject_dirs.append(item)
        
        subject_dirs = sorted(set(subject_dirs))
        print(f"AZDualCamera: 找到 {len(subject_dirs)} 个受试者目录")

        dirs = {}
        for subject_id in subject_dirs:
            subject_path = os.path.join(self.raw_data_path, subject_id)
            
            # 查找 V001, V002 等访问目录（可能是 V001 或 _V001）
            visit_dirs = []
            for item in os.listdir(subject_path):
                item_full = os.path.join(subject_path, item)
                if os.path.isdir(item_full) and (item.startswith('V') or item.startswith('_V')):
                    visit_dirs.append(item)
            visit_dirs = sorted(visit_dirs)
            
            if not visit_dirs:
                print(f"  ⚠️ {subject_id}: 未找到访问目录，跳过")
                continue
            
            for visit_name in visit_dirs:
                visit_path = os.path.join(subject_path, visit_name)
                visit_key = f"{subject_id}_{visit_name}"
                
                # 检查必要文件
                face_video = os.path.join(visit_path, "Face.mp4")
                finger_video = os.path.join(visit_path, "Finger.mp4")
                timestamp_csv = os.path.join(visit_path, "Timestamp.csv")
                ppg_csv = os.path.join(visit_path, "PPG_GT.csv")
                metrics_txt = os.path.join(visit_path, "Metrics.txt")
                
                if not os.path.exists(face_video):
                    print(f"  ⚠️ {visit_key}: 缺少 Face.mp4，跳过")
                    continue
                if not os.path.exists(ppg_csv):
                    print(f"  ⚠️ {visit_key}: 缺少 PPG_GT.csv，跳过")
                    continue
                
                # 解析 Metrics.txt 获取 BP 等信息
                bp_sys, bp_dia = None, None
                if os.path.exists(metrics_txt):
                    with open(metrics_txt, 'r', encoding='utf-8') as f:
                        content = f.read()
                    bp_match = re.search(r'BP:\s*(\d+)/(\d+)', content)
                    if bp_match:
                        bp_sys = int(bp_match.group(1))
                        bp_dia = int(bp_match.group(2))
                
                dirs[visit_key] = {
                    "face_video_path": face_video,
                    "finger_video_path": finger_video if os.path.exists(finger_video) else None,
                    "timestamp_path": timestamp_csv,
                    "ppg_path": ppg_csv,
                    "metrics_path": metrics_txt,
                    "bp_sys": bp_sys,
                    "bp_dia": bp_dia,
                    "subject_id": subject_id,
                    "visit_name": visit_name,
                }
        
        print(f"AZDualCamera: 成功加载 {len(dirs)} 个有效会话")
        return dirs

    def load_video_data(self, user_id=1):
        """
        Loads and processes video data for a given user.

        Args:
            user_id (int/str): The user ID for which to load the video data.

        Returns:
            pd.DataFrame: Merged video frames from front and back cameras.
        """
        user_id = str(user_id)
        
        if self.dataset_name == "Lab_multimodal":
            return self._load_lab_video_data(user_id)
        elif self.dataset_name == "AZDualCamera":
            return self._load_az_video_data(user_id)
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    def _load_lab_video_data(self, user_id):
        """加载 Lab_multimodal 数据集的视频数据"""
        videos = {}
        if self.use_front:
            front_video = self.read_video_sync(self.dirs[user_id]["front_camera_video_path"], video_type="Front")
            if self.front_standard_type == "Standardized":
                front_video = self._standardized_frame(front_video)
            elif self.front_standard_type == "diff_normalize":
                front_video = self._diff_normalize_frame(front_video)
            elif self.front_standard_type in ("raw", "facephys"):
                front_video = self._raw_normalize_frame(front_video)
            videos['front_video'] = front_video

        if self.use_back:
            back_video = self.read_video_sync(self.dirs[user_id]["back_camera_video_path"], video_type="Back")
            if self.back_standard_type == "Standardized":
                back_video = self._standardized_frame(back_video)
            elif self.back_standard_type == "diff_normalize":
                back_video = self._diff_normalize_frame(back_video)
            elif self.back_standard_type in ("raw", "facephys"):
                back_video = self._raw_normalize_frame(back_video)
            videos['back_video'] = back_video

        if len(videos) == 1:
            return self._handle_single_video(videos, user_id)
        else:
            return self.merge_video_frame(**videos, 
                                       front_video_timestamp_path=self.dirs[user_id]["front_camera_data_path"],
                                       back_video_timestamp_path=self.dirs[user_id]["back_camera_data_path"])

    def _load_az_video_data(self, user_id):
        """加载 AZDualCamera 数据集的视频数据"""
        user_info = self.dirs[user_id]
        videos = {}
        
        if self.use_front:
            front_video = self.read_video_sync(user_info["face_video_path"], video_type="Front")
            if self.front_standard_type == "Standardized":
                front_video = self._standardized_frame(front_video)
            elif self.front_standard_type == "diff_normalize":
                front_video = self._diff_normalize_frame(front_video)
            elif self.front_standard_type in ("raw", "facephys"):
                front_video = self._raw_normalize_frame(front_video)
            videos['front_video'] = front_video

        if self.use_back and user_info["finger_video_path"] is not None:
            back_video = self.read_video_sync(user_info["finger_video_path"], video_type="Back")
            if self.back_standard_type == "Standardized":
                back_video = self._standardized_frame(back_video)
            elif self.back_standard_type == "diff_normalize":
                back_video = self._diff_normalize_frame(back_video)
            elif self.back_standard_type in ("raw", "facephys"):
                back_video = self._raw_normalize_frame(back_video)
            videos['back_video'] = back_video

        if len(videos) == 1:
            return self._handle_single_video(videos, user_id)
        else:
            return self.merge_video_frame(**videos,
                                       front_video_timestamp_path=user_info["timestamp_path"],
                                       back_video_timestamp_path=user_info["timestamp_path"])
    
    def _handle_single_video(self, videos, user_id):
        """Handle the case of a single video modality."""
        video_type = 'front' if 'front_video' in videos else 'back'
        timestamp_path = self.dirs[f"{user_id}"][f"{video_type}_camera_data_path"]
        timestamps = self.load_video_timestamp(timestamp_path)['timestamp']
        video_data = videos[video_type+"_video"]
        min_length = min(len(timestamps), len(video_data))
        return pd.DataFrame({
            'timestamp': timestamps[:min_length],
            f'{video_type}_video': [frame for frame in video_data[:min_length]]
        })

    def _process_frames_sync(self, frame, process_flag, box=None, box_=None):
        """Process frames using face detection. Falls back to center-crop if no model available."""
        if not self._mp_facemesh_available:
            # Fallback: center-crop a square region
            h, w = frame.shape[:2]
            if process_flag:
                size = min(h, w)
                x0 = (w - size) // 2
                y0 = (h - size) // 2
                result = cv2.resize(frame[y0:y0+size, x0:x0+size],
                                    (self.target_weight, self.target_height),
                                    interpolation=cv2.INTER_AREA)
                return result, None, None
            else:
                if box_ is not None:
                    result = cv2.resize(frame[slice(*box_[1]), slice(*box_[0])],
                                        (self.target_weight, self.target_height),
                                        interpolation=cv2.INTER_AREA)
                    return result, box, box_
                size = min(h, w)
                x0 = (w - size) // 2
                y0 = (h - size) // 2
                result = cv2.resize(frame[y0:y0+size, x0:x0+size],
                                    (self.target_weight, self.target_height),
                                    interpolation=cv2.INTER_AREA)
                return result, box, box_

        if self._mp_facemesh is None:
            self._mp_facemesh = mp_vision.FaceLandmarker.create_from_options(self._mp_facemesh_options)

        h, w, c = frame.shape
        if w > h:
            frame_ = frame[:, round((w - h) / 2):round((w - h) / 2) + h]
        else:
            frame_ = frame
            w = h

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_)
        results = self._mp_facemesh.detect(mp_image)
        if process_flag:
            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmark = np.array([(p.x * h / w + round((w - h)/2)/w, p.y)
                                for p in results.face_landmarks[0]])
                shape = alphashape.alphashape(landmark, 0)
                if box is None:
                    box = np.array(shape.bounds).reshape(2, 2)
                else:
                    w_ema = 1/(1 + np.exp(-20*np.linalg.norm(np.array(shape.bounds).reshape(2, 2)-box)/np.multiply(*np.abs(box[0]-box[1]))))*2-1
                    box = np.array(shape.bounds).reshape(2, 2)*w_ema+box*(1-w_ema)
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
    
    def read_video_sync(self,video_path,video_type="Front", chunk_size=100):
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
            raise ValueError("Cannot open video file")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        """
        Calculate the discrete difference between consecutive frames along the time-axis and normalize the result by its standard deviation.

        Args:
            frames (numpy.ndarray): Input video frames with shape (n, h, w, c).

        Returns:
            numpy.ndarray: Normalized difference frames with shape (n, h, w, c).
        """
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
    
    def _raw_normalize_frame(self, frames):
        """FacePhys-compatible: uint8 → float32 [0, 1]. Model's TNM handles internal normalization."""
        frames = frames.astype(np.float32) / 255.0
        frames[np.isnan(frames)] = 0.0
        return frames

    @staticmethod
    def _norm_bvp_label(bvp):
        """FacePhys norm_bvp: iterative peak-based per-cycle normalization to [-1,1] → z-score.

        Ported from /root/zt/FacePhys/utils.py norm_bvp().
        """
        from scipy.signal import find_peaks
        import numpy as np

        def _norm(bvp_inner, order=1):
            order -= 1
            bvp_ = []
            _ = np.nan
            for v in bvp_inner:
                if np.isnan(v):
                    bvp_.append(_)
                else:
                    bvp_.append(v)
                    _ = v
            if np.isnan(bvp_[0]):
                for v in bvp_:
                    if not np.isnan(v):
                        _ = v
                        break
                n = 0
                while True:
                    if n >= len(bvp_) or not np.isnan(bvp_[0]):
                        break
                    bvp_[n] = _
                    n += 1
            bvp_ = np.array(bvp_)
            bvp_ = (bvp_ - np.mean(bvp_)) / np.std(bvp_)
            prominence = (1.5 * np.std(bvp_), None)
            peaks_q = find_peaks(bvp_, prominence=prominence)[0]
            peaks_v = find_peaks(-bvp_, prominence=prominence)[0]
            peaks = np.sort(np.concatenate([peaks_q, peaks_v]))
            if len(peaks) < 2:
                return (bvp_ - np.mean(bvp_)) / (np.std(bvp_) + 1e-6)
            bvp_ext = np.concatenate((bvp_, [bvp_[-1]]))
            boundaries = np.concatenate([(0,), peaks, (len(bvp_ext),)])
            segments = []
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                seg = bvp_ext[a:b]
                if len(seg) > 0:
                    seg_norm = ((seg - (np.max(seg) + np.min(seg)) / 2) / 
                                (np.max(seg) - np.min(seg) + 1e-8))
                    segments.append(seg_norm)
            if not segments:
                return (bvp_ - np.mean(bvp_)) / (np.std(bvp_) + 1e-6)
            bvp_norm = np.concatenate(segments)[:len(bvp_)]
            bvp_norm = np.clip(bvp_norm, -1.0, 1.0)
            if order == 0:
                return (bvp_norm - np.mean(bvp_norm)) / (np.std(bvp_norm) + 1e-6)
            else:
                return _norm(bvp_norm, order)
        return _norm(bvp, order=1)
    
    def _standardized_label(self, label):
        """
        Applies Z-score standardization to the label data.

        Args:
            label (pd.DataFrame): Input label data with 'timestamp' and other columns.

        Returns:
            pd.DataFrame: DataFrame with standardized values, preserving the timestamp column.
        """
        timestamp_col = label['timestamp']
        numeric_df = label.drop('timestamp', axis=1)
        standardized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
        standardized_df = standardized_df.fillna(0)
        standardized_df[np.isnan(standardized_df)] = 0
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
        timestamp_col = label['timestamp']
        numeric_df = label.drop('timestamp', axis=1)
        diff_df = numeric_df.diff()
        normalized_df = diff_df / diff_df.std()
        normalized_df = normalized_df.fillna(0)
        normalized_df[np.isnan(normalized_df)] = 0
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
        front_video_timestamp = self.load_video_timestamp(front_video_timestamp_path)
        back_video_timestamp = self.load_video_timestamp(back_video_timestamp_path)
        
        # 使用 wall_ms 作为时间戳进行对齐
        front_timestamp = front_video_timestamp['wall_ms'].values
        front_min_length = min(front_video.shape[0], len(front_timestamp))
        front_timestamp = front_timestamp[:front_min_length]
        front_video = front_video[:front_min_length]
        front_video_total = pd.DataFrame({
            'timestamp': front_timestamp,
            'front_video': [frame for frame in front_video]
        })
        
        back_timestamp = back_video_timestamp['wall_ms'].values
        back_min_length = min(back_video.shape[0], len(back_timestamp))
        back_timestamp = back_timestamp[:back_min_length]
        back_video = back_video[:back_min_length]
        back_video_total = pd.DataFrame({
            'timestamp': back_timestamp,
            'back_video': [frame for frame in back_video]
        })
        
        aligned_data = pd.merge_asof(front_video_total, back_video_total, on='timestamp', direction='nearest', suffixes=('_front', '_back'))
        return aligned_data

    def load_video_timestamp(self, video_timestamp_path):
        """
        Loads video timestamps from a CSV file.

        Args:
            video_timestamp_path (str): Path to the timestamp file.

        Returns:
            pd.DataFrame: A DataFrame containing the timestamps and corresponding frame numbers.
        """
        # 检测文件格式: 读取第一行判断是否有表头
        with open(video_timestamp_path, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('wall_ms'):
            # AZDualCamera 格式: wall_ms,sensor_ts_sec,relative_sec,frame_number
            df = pd.read_csv(video_timestamp_path)
            # 将 wall_ms 从毫秒转换为秒，与标签时间戳对齐
            df['wall_ms'] = df['wall_ms'] / 1000.0
        else:
            # Lab_multimodal 格式: timestamp,frame (无表头)
            df = pd.read_csv(video_timestamp_path, header=None, names=['timestamp', 'frame'])
            df = df.iloc[:-1, :]
        return df

    def load_label_data(self, user_id=1):
        """
        Loads label data for a given user.

        Args:
            user_id (int/str): The user ID for which to load the label data.

        Returns:
            pd.DataFrame: The loaded label data.
        """
        user_id = str(user_id)
        
        if self.dataset_name == "Lab_multimodal":
            return self._load_lab_label_data(user_id)
        elif self.dataset_name == "AZDualCamera":
            return self._load_az_label_data(user_id)
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    def _load_lab_label_data(self, user_id):
        """加载 Lab_multimodal 数据集的标签数据"""
        loaded_label_data = {}
        for label in self.labels:
            label_path = self.dirs[user_id][f"label_{label.upper()}_path"]
            temp_data = pd.read_csv(label_path)
            if self.label_standard_type == "Standardized":
                loaded_label_data[label] = self._standardized_label(temp_data)
            elif self.label_standard_type == "diff_normalize":
                loaded_label_data[label] = self._diff_normalize_label(temp_data)
            elif self.label_standard_type in ("raw", "facephys") and label == "bvp":
                bvp_raw = temp_data['bvp'].values if 'bvp' in temp_data.columns else temp_data.iloc[:, -1].values
                bvp_norm = self._norm_bvp_label(bvp_raw)
                loaded_label_data[label] = pd.DataFrame({
                    'timestamp': temp_data['timestamp'].values if 'timestamp' in temp_data.columns else temp_data.iloc[:, 0].values,
                    'bvp': bvp_norm
                })
            else:
                loaded_label_data[label] = temp_data
        label = self.merge_labels_data(Phys_indicators=loaded_label_data)
        loaded_label_data = None
        return label

    def _load_az_label_data(self, user_id):
        """
        加载 AZDualCamera 数据集的标签数据。
        
        标签来源:
        - PPG_GT.csv: BVP (timestamp_ms, bvp) - 血氧仪PPG信号
        - Metrics.txt: BP (血压收缩压/舒张压)
        - 从 BVP 信号计算 HR, RR, SpO2, HRV
        
        PPG_GT.csv 采样率: ~21Hz (唯一时间戳), 原始数据 ~72Hz (含重复时间戳)
        """
        user_info = self.dirs[user_id]
        loaded_label_data = {}
        
        # 1. 加载 PPG_GT.csv (BVP 信号)
        ppg_df = pd.read_csv(user_info["ppg_path"])
        ppg_df = ppg_df.rename(columns={"timestamp_ms": "timestamp"})
        # 将毫秒时间戳转换为秒 (与视频时间戳对齐)
        ppg_df['timestamp'] = ppg_df['timestamp'] / 1000.0
        # 去重：相同时间戳取平均
        ppg_df = ppg_df.groupby('timestamp', as_index=False).mean()
        ppg_df = ppg_df.sort_values('timestamp').reset_index(drop=True)
        
        if self.label_standard_type == "Standardized":
            loaded_label_data["bvp"] = self._standardized_label(ppg_df)
        elif self.label_standard_type == "diff_normalize":
            loaded_label_data["bvp"] = self._diff_normalize_label(ppg_df)
        elif self.label_standard_type in ("raw", "facephys"):
            # FacePhys norm_bvp: peak-based per-cycle normalization
            bvp_raw = ppg_df['bvp'].values
            bvp_norm = self._norm_bvp_label(bvp_raw)
            loaded_label_data["bvp"] = pd.DataFrame({
                'timestamp': ppg_df['timestamp'].values,
                'bvp': bvp_norm
            })
        else:
            loaded_label_data["bvp"] = ppg_df
        
        # 2. 从 BVP 信号计算 HR (心率)
        bvp_values = ppg_df['bvp'].values
        timestamps = ppg_df['timestamp'].values
        
        # 计算时间差 (秒) - timestamps 已经是秒
        time_diffs = np.diff(timestamps)
        median_dt = np.median(time_diffs) if len(time_diffs) > 0 else 0.05
        
        from scipy.signal import find_peaks, butter, filtfilt
        
        # 带通滤波提取心率分量 (0.8-3.0 Hz 对应 48-180 bpm)
        nyquist = 0.5 / median_dt
        b_hr, a_hr = butter(2, [0.8 / nyquist, 3.0 / nyquist], btype='band')
        bvp_filtered = filtfilt(b_hr, a_hr, bvp_values)
        
        # 全局峰值检测
        min_distance = int(0.3 / median_dt)  # 最小间隔 0.3s (200 bpm)
        peaks, _ = find_peaks(bvp_filtered, distance=min_distance)
        
        # 计算瞬时心率
        hr_values = np.full(len(bvp_values), np.nan)
        if len(peaks) > 1:
            peak_times = timestamps[peaks]
            for i in range(len(bvp_values)):
                # 找到当前时间点前后的两个峰值
                prev_peaks = peak_times[peak_times <= timestamps[i]]
                next_peaks = peak_times[peak_times > timestamps[i]]
                if len(prev_peaks) >= 2:
                    # 使用前两个峰值的间隔
                    interval = prev_peaks[-1] - prev_peaks[-2]
                    if 0.3 < interval < 1.5:  # 40-200 bpm
                        hr_values[i] = 60.0 / interval
                elif len(prev_peaks) >= 1 and len(next_peaks) >= 1:
                    interval = next_peaks[0] - prev_peaks[-1]
                    if 0.3 < interval < 1.5:
                        hr_values[i] = 60.0 / interval
        
        # 填充 NaN
        hr_series = pd.Series(hr_values).ffill().bfill()
        if hr_series.isna().all():
            # 如果全部 NaN，使用全局平均
            if len(peaks) > 1:
                global_hr = 60.0 / np.mean(np.diff(peak_times))
                hr_series = pd.Series(np.full(len(bvp_values), global_hr))
            else:
                hr_series = pd.Series(np.full(len(bvp_values), 75.0))
        
        hr_df = pd.DataFrame({
            'timestamp': timestamps,
            'hr': hr_series.values
        })
        
        if self.label_standard_type == "Standardized":
            loaded_label_data["hr"] = self._standardized_label(hr_df)
        elif self.label_standard_type == "diff_normalize":
            loaded_label_data["hr"] = self._diff_normalize_label(hr_df)
        else:
            loaded_label_data["hr"] = hr_df
        
        # 3. 从 BVP 信号计算 RR (呼吸率) - 使用包络检测
        from scipy.signal import butter, filtfilt
        
        # 带通滤波提取呼吸分量 (0.1-0.5 Hz)
        nyquist = 0.5 / median_dt
        b_rr, a_rr = butter(2, [0.1 / nyquist, 0.5 / nyquist], btype='band')
        bvp_rr = filtfilt(b_rr, a_rr, bvp_values)
        
        # 呼吸峰值检测
        min_rr_distance = int(1.0 / median_dt)  # 最小间隔 1s (60 rpm)
        rr_peaks, _ = find_peaks(bvp_rr, distance=min_rr_distance)
        
        rr_values = np.full(len(bvp_values), np.nan)
        if len(rr_peaks) > 1:
            rr_peak_times = timestamps[rr_peaks]
            for i in range(len(bvp_values)):
                prev_peaks = rr_peak_times[rr_peak_times <= timestamps[i]]
                if len(prev_peaks) >= 2:
                    interval = prev_peaks[-1] - prev_peaks[-2]
                    if 1.5 < interval < 10.0:  # 6-40 rpm
                        rr_values[i] = 60.0 / interval
        
        rr_series = pd.Series(rr_values).ffill().bfill()
        if rr_series.isna().all():
            if len(rr_peaks) > 1:
                global_rr = 60.0 / np.mean(np.diff(rr_peak_times))
                rr_series = pd.Series(np.full(len(bvp_values), global_rr))
            else:
                rr_series = pd.Series(np.full(len(bvp_values), 18.0))
        
        rr_df = pd.DataFrame({
            'timestamp': timestamps,
            'rr': rr_series.values
        })
        
        if self.label_standard_type == "Standardized":
            loaded_label_data["rr"] = self._standardized_label(rr_df)
        elif self.label_standard_type == "diff_normalize":
            loaded_label_data["rr"] = self._diff_normalize_label(rr_df)
        else:
            loaded_label_data["rr"] = rr_df
        
        # 4. SpO2 (血氧) - 从 BVP 信号幅度估算
        spo2_values = np.full(len(bvp_values), np.nan)
        window_samples = int(6.0 / median_dt)  # 6秒窗口
        if len(bvp_values) > window_samples:
            for i in range(0, len(bvp_values), max(1, window_samples // 2)):
                start = max(0, i - window_samples // 2)
                end = min(len(bvp_values), i + window_samples // 2)
                segment = bvp_values[start:end]
                if len(segment) > 10:
                    ac = np.std(segment)
                    dc = np.mean(segment)
                    if dc > 0:
                        ratio = ac / dc
                        spo2 = 100.0 - 20.0 * ratio
                        spo2 = np.clip(spo2, 85, 100)
                        spo2_values[i] = spo2
        
        spo2_series = pd.Series(spo2_values).ffill().bfill().fillna(97.0)
        
        spo2_df = pd.DataFrame({
            'timestamp': timestamps,
            'spo2': spo2_series.values
        })
        
        if self.label_standard_type == "Standardized":
            loaded_label_data["spo2"] = self._standardized_label(spo2_df)
        elif self.label_standard_type == "diff_normalize":
            loaded_label_data["spo2"] = self._diff_normalize_label(spo2_df)
        else:
            loaded_label_data["spo2"] = spo2_df
        
        # 5. BP (血压) - 从 Metrics.txt 获取 (静态值，不做 diff_normalize)
        bp_sys = user_info.get("bp_sys")
        bp_dia = user_info.get("bp_dia")
        
        if bp_sys is not None and bp_dia is not None:
            bp_sys_df = pd.DataFrame({
                'timestamp': timestamps,
                'bp_sys': float(bp_sys)
            })
            bp_dia_df = pd.DataFrame({
                'timestamp': timestamps,
                'bp_dia': float(bp_dia)
            })
            
            # BP 是静态值，只做 Standardized 或保持原值
            if self.label_standard_type == "Standardized":
                loaded_label_data["bp_sys"] = self._standardized_label(bp_sys_df)
                loaded_label_data["bp_dia"] = self._standardized_label(bp_dia_df)
            else:
                # diff_normalize 对静态值无效，保持原值
                loaded_label_data["bp_sys"] = bp_sys_df
                loaded_label_data["bp_dia"] = bp_dia_df
        
        # 合并所有标签 - 使用浮点数时间戳（秒），不经过 datetime 转换
        # 找到所有标签的时间戳范围
        all_timestamps = []
        for label_name, df in loaded_label_data.items():
            all_timestamps.append(df['timestamp'].values)
        all_timestamps = np.concatenate(all_timestamps)
        ts_min, ts_max = all_timestamps.min(), all_timestamps.max()
        
        # 生成目标时间戳 (30Hz)
        target_ts = np.arange(ts_min, ts_max, 1.0 / self.target_Hz)
        target_df = pd.DataFrame({'timestamp': target_ts})
        
        # 对每个标签进行重采样
        resampled_list = []
        for label_name, df in loaded_label_data.items():
            resampled = pd.merge_asof(target_df, df, on='timestamp', direction='nearest')
            resampled_list.append(resampled)
        
        # 合并所有标签
        label = resampled_list[0]
        for df in resampled_list[1:]:
            label = pd.merge(label, df, on='timestamp', how='outer')
        
        label = label.sort_values('timestamp').reset_index(drop=True)
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
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'],unit='s')
        start_time = raw_data['timestamp'].min()
        end_time = raw_data['timestamp'].max()
        if target_length is not None:
            new_timestamps = pd.date_range(start=start_time, end=end_time, periods=target_length)
        else:
            new_timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{1000 // target_Hz}ms')
        new_timestamps_df = pd.DataFrame(new_timestamps, columns=['timestamp'])
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
        resampled_data_list = []
        for label, data in Phys_indicators.items():
            resampled_data = self.resample_data(data, target_Hz, target_length)
            resampled_data_list.append(resampled_data)
        if len(resampled_data_list) == 1:
            aligned_data = resampled_data_list[0]
        else:
            aligned_data = resampled_data_list[0]
            for data in resampled_data_list[1:]:
                aligned_data = pd.merge_asof(aligned_data, data, on='timestamp', direction='nearest')
        aligned_data['timestamp'] = pd.to_datetime(aligned_data['timestamp'])
        return aligned_data

    def load_merged_all_data(self, user_id=1):
        """
        Loads and merges all data (video and labels) for a given user.

        Args:
            user_id (int/str): The user ID for which to load the data. Default is 1.

        Returns:
            pd.DataFrame: The merged data containing the video and label data of target user.
        """
        user_id = str(user_id)
        print(f"\033[91mUser_{user_id}_data_loading_start\033[0m")
        video_data = self.load_video_data(user_id)
        label_data = self.load_label_data(user_id)
        
        if self.dataset_name in ["Lab_multimodal", "AZDualCamera"]:
            all_data = pd.merge_asof(video_data, label_data, on='timestamp', direction='nearest')
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        
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
        train_dataset.save_dataset_to_hdf("/data02/Lab/zt/Dataset_pt/train_dataset.h5")
        test_dataset.save_dataset_to_hdf("/data02/Lab/zt/Dataset_pt/test_dataset.h5")

    def _split_into_sequences(self, data, stride=None):
        """
        Splits the input data into sequences of a fixed length.

        Args:
            data (torch.Tensor): The input data to be split into sequences.
            stride (int): stride between sequence starts. Default = seq_len (non-overlapping).

        Returns:
            torch.Tensor: A tensor containing the sequences of data.
        """
        if stride is None:
            stride = self.seq_len
        num_sequences = (data.shape[0] - self.seq_len) // stride + 1
        sequences = [data[i * stride:i * stride + self.seq_len] for i in range(num_sequences)]
        return torch.stack(sequences)

    def _apply_processing(self, data, processing_type, processing_functions):
        """Applies a specified processing function to each sample in the input data."""
        if processing_type in processing_functions:
            processed_data = []
            for sample in data:
                # 将 tensor 转为 numpy 进行处理，再转回 tensor
                if torch.is_tensor(sample):
                    sample_np = sample.cpu().numpy()
                    processed_np = processing_functions[processing_type](sample_np)
                    processed_data.append(torch.from_numpy(processed_np))
                else:
                    processed_sample = processing_functions[processing_type](sample)
                    processed_data.append(processed_sample)
            return torch.stack(processed_data)
        return data

    def _create_dataset(self, data):
        """
        Creates a dataset from the provided data by converting it into tensors.
        
        NOTE: Normalization is already applied in load_video_data / load_label_data.
        _apply_processing is intentionally NOT called here to avoid double-processing.
        """
        features = {}
        if self.use_front:
            features["video_front"] = torch.tensor(np.stack(data["front_video"]), dtype=torch.float32)
            features["video_front"] = self._split_into_sequences(features["video_front"])
        if self.use_back:
            features["video_back"] = torch.tensor(np.stack(data["back_video"]), dtype=torch.float32)
            features["video_back"] = self._split_into_sequences(features["video_back"])           
        if self.dataset_name == "Lab_multimodal":
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
        elif self.dataset_name == "AZDualCamera":
            # 动态构建标签张量 - 只包含数据中存在的列
            available_cols = [col for col in ["bvp", "hr", "rr", "spo2", "bp_sys", "bp_dia"] if col in data.columns]
            label_values = data[available_cols].values
            labels = self._split_into_sequences(torch.tensor(label_values, dtype=torch.float32))
            label_dict = {col: labels[:,:,i] for i, col in enumerate(available_cols)}
            return MultimodalDataset(
                features=features,
                labels=label_dict,
                use_front=self.use_front,
                use_back=self.use_back
            )
        
    def save_datasets(self, path):
        """Save the processed datasets for each user to the specified directory.

        Args:
            path (str): The directory path where the datasets will be saved.
        """
        os.makedirs(path, exist_ok=True)
        user_ids = list(self.dirs.keys())
        for user_id in user_ids:
            try:
                dataset_df = self.load_merged_all_data(user_id)
                dataset = self._create_dataset(dataset_df)
                save_path = f"{path}/{self.dataset_name}_{self.front_standard_type}_label_{self.label_standard_type}_dataset_{user_id}.pth"
                torch.save(dataset, save_path, pickle_protocol=4)
                print(f"  ✅ 已保存: {save_path}")
            except Exception as e:
                print(f"  ❌ 用户 {user_id} 处理失败: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # ============================================
    # 配置参数说明
    # ============================================
    # 数据集参数:
    #   dataset_name: "Lab_multimodal" | "AZDualCamera"
    #   data_path: 数据集根目录路径
    #   
    # 视频处理参数:
    #   resize_height: 128  (视频帧resize高度)
    #   resize_width: 128   (视频帧resize宽度)
    #   target_hz: 30       (目标采样率)
    #   seq_len: 160        (序列长度, 对应 ~5.3秒 @30Hz)
    #   use_front: True     (是否使用前置摄像头)
    #   use_back: True      (是否使用后置摄像头)
    #   
    # 标准化参数:
    #   front_standard_type: "diff_normalize" | "Standardized" | None
    #   back_standard_type: "diff_normalize" | "Standardized" | None
    #   label_standard_type: "diff_normalize" | "Standardized" | None
    #
    # AZDualCamera 数据集信息:
    #   - Face.mp4: 前置摄像头, 1920x1080, 30fps, ~59秒
    #   - Finger.mp4: 后置摄像头, 1920x1080, 30fps, ~59秒
    #   - PPG_GT.csv: BVP信号, ~21Hz (唯一时间戳)
    #   - Metrics.txt: 含 BP (血压), Age, Gender 等信息
    #   - 标签: bvp, hr (从BVP计算), rr (从BVP计算), spo2 (估算), bp_sys, bp_dia
    # ============================================
    
    # === 示例1: 处理 Lab_multimodal 数据集 ===
    # config = {
    #     "dataset_name": "Lab_multimodal",
    #     "data_path": "F3mamba/Dataset/LabDataset",
    #     "front_standard_type": "diff_normalize",
    #     "back_standard_type": "diff_normalize",
    #     "label_standard_type": "diff_normalize",
    #     "resize_height": 128,
    #     "resize_width": 128,
    #     "target_hz": 30,
    #     "seq_len": 160,
    #     "use_front": True,
    #     "use_back": True,
    # }
    
    # === 示例2: 处理 AZDualCamera 数据集 (FacePhys 对齐) ===
    config = {
        "dataset_name": "AZDualCamera",
        "data_path": "/root/shared/az_dual_camera/az_hspt_data/data/organized",
        "front_standard_type": "facephys",      # uint8→/255, TNM handles internal norm
        "back_standard_type": "facephys",
        "label_standard_type": "facephys",       # norm_bvp peak-based BVP normalization
        "resize_height": 36,                     # FacePhys InfinitePulse uses 36×36
        "resize_width": 36,
        "target_hz": 30,
        "seq_len": 160,
        "use_front": True,
        "use_back": True,
    }
    
    # 将 dict 转换为 object 以兼容现有代码
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config_obj = Config(config)
    data_loader = MultimodalDataLoader(config_obj)
    data_loader.save_datasets("./ProcessedDataset")
