import numpy as np 
import pandas as pd
import os
import sys
import time
import cv2
import glob
from tqdm import tqdm
import torch
from queue import Queue
import alphashape
import mediapipe as mp
import math

             
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
        
        # Add available modalities to sample
        for modality in ["video_front", "video_back"]:
            if modality in self.data:
                sample["modals"][modality] = self.data[modality][idx]

        for modality in ["bvp", "hr", "rr", "spo2"]:
            if modality in self.data["labels"]:
                sample["labels"][modality] = self.data["labels"][modality][idx]

        return sample


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
        self.label_standard_type = config.label_standard_type
        
        # resize config
        self.target_height = config.resize_height
        self.target_weight = config.resize_width
        self.target_Hz = config.target_hz
        self.seq_len = config.seq_len

        # test config
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
        # 排除 id 4（视频错误）, 2（时间太短）, 5 and 6 （时间戳无法对齐）, 16 (weith problem)
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
            front_video = self.read_video_sync(self.dirs[f"{user_id}"]["front_camera_video_path"])
            if self.front_standard_type == "Standardized":
                front_video = self._standardized_frame(front_video)
            elif self.front_standard_type == "diff_normalize":
                front_video = self._diff_normalize_frame(front_video)
            videos['front_video'] = front_video

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
    
    def read_video_sync(self,video_path,video_type="Front", chunk_size=100):
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
        """
        Calculate the discrete difference between consecutive frames along the time-axis and normalize the result by its standard deviation.

        Args:
            frames (numpy.ndarray): Input video frames with shape (n, h, w, c), where:
                - n: Number of frames.
                - h: Height of each frame.
                - w: Width of each frame.
                - c: Number of channels.

        Returns:
            numpy.ndarray: Normalized difference frames with shape (n, h, w, c), where:
                - The last frame is padded with zeros to maintain the original frame count.
                - NaN values are replaced with zeros.
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
        Loads and merges all data (video and labels) for a given user.

        Args:
            user_id (int): The user ID for which to load the data. Default is 1.

        Returns:
            pd.DataFrame: The merged data containing the video and label data of target user.
            Keys: ['timestamp', 'front_video', 'back_video', 'bvp', 'hr', 'rr', 'spo2']
        """
        # Step 1: Load video data and label data
        print(f"\033[91mUser_{user_id}_data_loading_start\033[0m")
        video_data = self.load_video_data(user_id)
        label_data = self.load_label_data(user_id)

        # Step 2: Merger features and label 
        if self.dataset_name == "tjk_multimodal":
            all_data = pd.merge_asof(video_data, label_data, on='timestamp', direction='nearest')
        elif self.dataset_name == "anzhen_multimodal":
            # calculate the diff timestamp
            timestamp_gap = label_data['timestamp'].iloc[0] - video_data['timestamp'].iloc[0]
            # add the gap to the video_data
            video_data['timestamp'] = video_data['timestamp'] + timestamp_gap
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

        # 保存 Dataset 对象
        train_dataset.save_dataset_to_hdf("/data02/tjk/zt/Dataset_pt/train_dataset.h5")
        test_dataset.save_dataset_to_hdf("/data02/tjk/zt/Dataset_pt/test_dataset.h5")

    def _create_dataset(self, data):
        """
        Creates a dataset from the provided data by converting it into tensors.

        Args:
            data (pd.DataFrame): The input data containing features and labels.

        Returns:
            MultimodalDataset: A dataset object containing the features and labels.
        """

        def _apply_processing(self, data, processing_type, processing_functions):
            """Applies a specified processing function to each sample in the input data.

            Args:
                data: Input data to be processed.
                processing_type: Key to identify the processing function in `processing_functions`.
                processing_functions: Dictionary mapping processing types to their respective functions.

            Returns:
                np.ndarray: Stacked processed data if `processing_type` is found, otherwise returns the original data.
            """
            if processing_type in processing_functions:
                processed_data = []
                for sample in data:
                    processed_sample = processing_functions[processing_type](sample)
                    processed_data.append(processed_sample)
                return np.stack(processed_data)
            return data

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
        
        features = {}
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
        
    def save_datasets(self,path):
        """Save the processed datasets for each user to the specified directory.

        Args:
            path (str): The directory path where the datasets will be saved.
        """
        user_ids = self.dirs.keys()
        for user_id in user_ids:
            dataset_df = self.load_merged_all_data(user_id)
            dataset = self._create_dataset(dataset_df)
            torch.save(dataset,f"{path}/video_{self.front_standard_type}_label_{self.label_standard_type}_dataset_{user_id}.pth",pickle_protocol=4)


if __name__ == "__main__":
    # Example usage
    config = {
        "data_path": "vPPG-Fusion/lab_raw_dataset",
        "front_standard_type": "diff_normalize",
        "back_standard_type": "diff_normalize",
        "label_standard_type": "diff_normalize",
        "resize_height": 128,
        "resize_width": 128,
        "target_hz": 30,
        "seq_len": 120,
    }
    
    data_loader = MultimodalDataLoader(config)
    data_loader.save_datasets("vPPG-Fusion/Dataset")