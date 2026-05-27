"""Preprocess 30 AZDualCamera subjects with FacePhys-aligned settings (36x36, facephys norm)."""
import sys, os
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

all_ids = sorted(loader.dirs.keys())
process_ids = all_ids[:30]
print(f'Total subjects: {len(all_ids)}, processing first {len(process_ids)}: {process_ids}')

original_dirs = loader.dirs
loader.dirs = {k: v for k, v in original_dirs.items() if k in process_ids}

output_dir = './ProcessedDataset/AZDualCamera'
os.makedirs(output_dir, exist_ok=True)
loader.save_datasets(output_dir)

print(f'\nDone! Processed {len(process_ids)} subjects.')
print(f'Output: {output_dir}')

import glob
pth_files = glob.glob(f'{output_dir}/AZDualCamera_front_facephys_label_facephys_dataset_*.pth')
print(f'Total facephys .pth files: {len(pth_files)}')
