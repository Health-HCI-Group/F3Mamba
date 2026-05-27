"""Re-process AZDualCamera with FacePhys-aligned settings."""
import sys, os, time
sys.path.insert(0, '.')
from Process.data_process import MultimodalDataLoader

config = {
    'dataset_name': 'AZDualCamera',
    'data_path': '/root/shared/az_dual_camera/az_hspt_data/data/organized',
    'front_standard_type': 'facephys',
    'back_standard_type': 'facephys',
    'label_standard_type': 'facephys',
    'resize_height': 36,
    'resize_width': 36,
    'target_hz': 30,
    'seq_len': 160,
    'use_front': True,
    'use_back': True,
}

class Config: 
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)

loader = MultimodalDataLoader(Config(config))

# Process first 10 subjects for training
all_ids = sorted(loader.dirs.keys())
process_ids = all_ids[:10]
print(f'Total subjects: {len(all_ids)}, processing {len(process_ids)}: {process_ids}')

# Override dirs to only process subset
original_dirs = loader.dirs
loader.dirs = {k: v for k, v in original_dirs.items() if k in process_ids}

output_dir = './ProcessedDataset/AZDualCamera'  # same dir, different filename pattern
loader.save_datasets(output_dir)
print(f'Done! Output: {output_dir}')
