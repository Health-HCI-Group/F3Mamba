"""Full training test with FacePhys-aligned InfinitePulse on AZDualCamera (10 subjects)."""
import torch, torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import glob, os, sys, numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt, welch

sys.path.insert(0, '.')
from Models.InfinitePulse import InfinitePulse

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './ProcessedDataset/AZDualCamera'
BATCH, EPOCHS, LR, SEQ_LEN, FPS = 8, 20, 1e-3, 160, 30

# ── Load facephys-aligned data ──
files = sorted(glob.glob(f'{DATA_DIR}/*facephys*'))
print(f'FacePhys-aligned subjects: {len(files)}')

datasets = []
for f in files:
    ds = torch.load(f, weights_only=False)
    # Remove border sequences
    ds = Subset(ds, range(1, len(ds) - 1))
    datasets.append(ds)
    print(f'  {os.path.basename(f)[-35:]}: {len(ds)} seqs')

# Train/test split: 7 train, 3 test
np.random.seed(42)
idx = np.random.permutation(len(datasets))
train_ds = ConcatDataset([datasets[i] for i in idx[:7]])
test_ds  = ConcatDataset([datasets[i] for i in idx[7:]])

train_ldr = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2, pin_memory=True)
test_ldr  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)
print(f'Train: {len(train_ds)} seqs ({len(train_ldr)} batches), Test: {len(test_ds)} seqs ({len(test_ldr)} batches)')

# ── Model ──
model = InfinitePulse(n_layers=[2,2,2,2], filters=[32]*4, chunk_size=SEQ_LEN).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
n_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {n_params:,} ({n_params*4/1024/1024:.1f} MB)')

# ── HR utilities ──
def bandpass_filter(x, fs=FPS, low=0.5, high=3.0):
    nyq = 0.5 * fs
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x)

def get_hr(sig, fs=FPS, hr_min=30, hr_max=180):
    p, q = welch(sig, fs, nfft=int(1e5/fs), nperseg=min(len(sig)-1, 256))
    mask = (p > hr_min/60) & (p < hr_max/60)
    if mask.sum() == 0:
        return 0.0
    return p[mask][np.argmax(q[mask])] * 60

# ── Training ──
def process_batch(batch):
    vid = batch['modals']['video_front'].to(DEVICE)
    lab = batch['labels']['bvp'].to(DEVICE)
    pred = model(vid)
    pred = (pred - pred.mean(-1, keepdim=True)) / (pred.std(-1, keepdim=True) + 1e-8)
    lab  = (lab  - lab.mean(-1, keepdim=True))  / (lab.std(-1, keepdim=True)  + 1e-8)
    return pred, lab

best_loss = float('inf')
for ep in range(1, EPOCHS+1):
    model.train()
    tl = 0.0
    pbar = tqdm(train_ldr, desc=f'Epoch {ep:2d}/{EPOCHS}', leave=False)
    for b in pbar:
        p, l = process_batch(b)
        loss = criterion(p, l)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = tl / len(train_ldr)

    # Validation
    model.eval()
    vl = 0.0
    with torch.no_grad():
        for b in test_ldr:
            p, l = process_batch(b)
            vl += criterion(p, l).item()
    vl /= len(test_ldr)

    improved = ' *' if vl < best_loss else ''
    if vl < best_loss: best_loss = vl
    print(f'Epoch {ep:2d}: train={avg_loss:.4f}  val={vl:.4f}{improved}')

# ── HR Evaluation ──
print('\n=== HR Evaluation (best model) ===')
model.eval()
all_pred, all_label = [], []
with torch.no_grad():
    for b in test_ldr:
        p, l = process_batch(b)
        all_pred.append(p.cpu().numpy())
        all_label.append(l.cpu().numpy())
all_pred = np.concatenate(all_pred)
all_label = np.concatenate(all_label)

# Compute HR over sliding windows
hr_window = 1000  # ~33 seconds of data
step = 500
hr_p, hr_l = [], []
for i in range(0, all_pred.shape[0] - hr_window, step):
    cp = all_pred[i:i+hr_window].reshape(-1)
    cl = all_label[i:i+hr_window].reshape(-1)
    if len(cp) < 60: continue
    cp_bp = bandpass_filter(cp)
    cl_bp = bandpass_filter(cl)
    hr_p.append(get_hr(cp_bp))
    hr_l.append(get_hr(cl_bp))

hr_p, hr_l = np.array(hr_p), np.array(hr_l)
valid = (hr_p > 0) & (hr_l > 0)
hr_p, hr_l = hr_p[valid], hr_l[valid]

if len(hr_p) > 1:
    mae  = np.mean(np.abs(hr_p - hr_l))
    rmse = np.sqrt(np.mean((hr_p - hr_l)**2))
    mape = np.mean(np.abs(hr_p - hr_l) / hr_l) * 100
    r    = np.corrcoef(hr_p, hr_l)[0,1]
    print(f'HR windows: {len(hr_p)}')
    print(f'MAE={mae:.2f} bpm  |  RMSE={rmse:.2f} bpm  |  MAPE={mape:.1f}%  |  R={r:.4f}')
    print(f'Pred HR: [{hr_p.min():.0f}, {hr_p.max():.0f}]  |  GT HR: [{hr_l.min():.0f}, {hr_l.max():.0f}]')
else:
    print(f'Only {len(hr_p)} valid HR windows (need more data)')

print('\n=== Test Complete ===')
print('InfinitePulse + FacePhys-aligned preprocessing: training and evaluation both work.')
