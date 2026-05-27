import csv
import json
import re
from pathlib import Path
from datetime import datetime


ROOT = Path('/root/zt/F3Mamba')
OUT_CSV = ROOT / 'docs/superpowers/specs/bp_master_table.csv'
OUT_MD = ROOT / 'docs/superpowers/specs/bp_master_table.md'


def as_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_result_jsons(rows):
    for p in sorted(ROOT.glob('train_*_result.json')):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        row = {
            'source': 'result_json',
            'id': p.stem,
            'status': 'completed' if bool(data.get('completed')) else 'incomplete',
            'bp_loss_type': None,
            'bp_huber_delta': None,
            'align_mode': None,
            'lr': None,
            'bp_dropout': None,
            'bvp_loss_weight': None,
            'recon_loss_weight': None,
            'avg_mae_sys': None,
            'avg_pearson_sys': None,
            'hr_mae': None,
            'hr_pearson': None,
            'note': str(p),
        }

        cfg = data.get('resolved_config') or data.get('config') or {}
        row['align_mode'] = cfg.get('align_mode')
        row['lr'] = as_float(cfg.get('lr'))
        row['bp_dropout'] = as_float(cfg.get('bp_dropout'))
        row['bvp_loss_weight'] = as_float(cfg.get('bvp_loss_weight'))
        row['recon_loss_weight'] = as_float(cfg.get('recon_loss_weight'))
        row['bp_loss_type'] = cfg.get('bp_loss_type')
        row['bp_huber_delta'] = as_float(cfg.get('bp_huber_delta'))

        row['avg_mae_sys'] = as_float(data.get('avg_mae_sys')) or as_float(data.get('bp_sys_avg'))
        row['avg_pearson_sys'] = as_float(data.get('avg_pearson_sys')) or as_float(data.get('bp_sys_pearson'))

        folds = data.get('fold_results') or []
        hr_mae_vals = [as_float(f.get('hr_mae')) for f in folds if isinstance(f, dict) and as_float(f.get('hr_mae')) is not None]
        hr_p_vals = [as_float(f.get('hr_pearson')) for f in folds if isinstance(f, dict) and as_float(f.get('hr_pearson')) is not None]
        if hr_mae_vals:
            row['hr_mae'] = sum(hr_mae_vals) / len(hr_mae_vals)
        if hr_p_vals:
            row['hr_pearson'] = sum(hr_p_vals) / len(hr_p_vals)

        rows.append(row)


def parse_robust_log(rows):
    log_path = ROOT / 'run_sweep_xmae_v3_bp_huber.log'
    if not log_path.exists():
        return

    text = log_path.read_text(errors='ignore')
    parts = re.split(r'Agent Starting Run: (\w+)', text)
    for i in range(1, len(parts), 2):
        run_id = parts[i]
        content = parts[i + 1]

        m_loss = re.search(r'loss: bp_loss_type=([^,\n]+), bp_huber_delta=([0-9.]+)', content)
        m_cfg = re.search(
            r'train: lr=([0-9.eE+-]+).*?model:.*?bp_dropout=([0-9.]+).*?recon_w=([0-9.]+), bvp_w=([0-9.]+)',
            content,
            flags=re.S,
        )
        m_align = re.search(r'align: mode=([^,\n]+)', content)
        m_final = re.search(r'Average MAE: ([0-9.]+), Pearson: (-?[0-9.]+)', content)

        if m_final is None:
            continue

        row = {
            'source': 'robust_log',
            'id': run_id,
            'status': 'completed',
            'bp_loss_type': m_loss.group(1).strip() if m_loss else None,
            'bp_huber_delta': as_float(m_loss.group(2)) if m_loss else None,
            'align_mode': m_align.group(1).strip() if m_align else None,
            'lr': as_float(m_cfg.group(1)) if m_cfg else None,
            'bp_dropout': as_float(m_cfg.group(2)) if m_cfg else None,
            'bvp_loss_weight': as_float(m_cfg.group(4)) if m_cfg else None,
            'recon_loss_weight': as_float(m_cfg.group(3)) if m_cfg else None,
            'avg_mae_sys': as_float(m_final.group(1)),
            'avg_pearson_sys': as_float(m_final.group(2)),
            'hr_mae': None,
            'hr_pearson': None,
            'note': str(log_path),
        }

        m_hr_mae = re.search(r'eval/hr_MAE\s+([0-9.]+)', content)
        m_hr_p = re.search(r'eval/hr_Pearson\s+([0-9.]+)', content)
        if m_hr_mae:
            row['hr_mae'] = as_float(m_hr_mae.group(1))
        if m_hr_p:
            row['hr_pearson'] = as_float(m_hr_p.group(1))

        rows.append(row)


def parse_markdown_rows(rows):
    md_path = ROOT / 'docs/superpowers/specs/2026-05-20-seismo-samsung-fusion-experiment-log.md'
    if not md_path.exists():
        return

    for line in md_path.read_text(errors='ignore').splitlines():
        if not line.strip().startswith('| `'):
            continue
        parts = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(parts) < 6:
            continue
        run = parts[0].strip('`')
        mae = as_float(parts[2])
        pear = as_float(parts[3])
        hr_mae = as_float(parts[4])
        hr_p = as_float(parts[5])
        if mae is None or pear is None:
            continue
        rows.append({
            'source': 'fusion_md',
            'id': run,
            'status': 'completed',
            'bp_loss_type': None,
            'bp_huber_delta': None,
            'align_mode': None,
            'lr': None,
            'bp_dropout': None,
            'bvp_loss_weight': None,
            'recon_loss_weight': None,
            'avg_mae_sys': mae,
            'avg_pearson_sys': pear,
            'hr_mae': hr_mae,
            'hr_pearson': hr_p,
            'note': str(md_path),
        })


def dedup(rows):
    by_id = {}
    source_rank = {'robust_log': 3, 'result_json': 2, 'fusion_md': 1}
    for r in rows:
        k = r['id']
        prev = by_id.get(k)
        if prev is None:
            by_id[k] = r
            continue
        if source_rank.get(r['source'], 0) >= source_rank.get(prev['source'], 0):
            merged = prev.copy()
            for key, val in r.items():
                if val is not None and val != '':
                    merged[key] = val
            by_id[k] = merged
    return list(by_id.values())


def write_outputs(rows):
    rows = sorted(rows, key=lambda r: (r['avg_pearson_sys'] is None, -(r['avg_pearson_sys'] or -999)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'source', 'id', 'status', 'bp_loss_type', 'bp_huber_delta', 'align_mode',
        'lr', 'bp_dropout', 'bvp_loss_weight', 'recon_loss_weight',
        'avg_mae_sys', 'avg_pearson_sys', 'hr_mae', 'hr_pearson', 'note',
    ]
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    top = [r for r in rows if r.get('avg_pearson_sys') is not None][:10]
    lines = [
        '# BP Master Table',
        '',
        f'- Generated at: {ts}',
        f'- Total rows: {len(rows)}',
        f'- CSV path: `{OUT_CSV}`',
        '',
        '## Top by avg_pearson_sys',
        '',
        '| Rank | ID | source | avg_mae_sys | avg_pearson_sys | hr_mae | bp_loss_type | delta |',
        '|---|---|---|---:|---:|---:|---|---:|',
    ]
    for i, r in enumerate(top, start=1):
        lines.append(
            f"| {i} | `{r['id']}` | {r['source']} | {r['avg_mae_sys'] if r['avg_mae_sys'] is not None else ''} | {r['avg_pearson_sys'] if r['avg_pearson_sys'] is not None else ''} | {r['hr_mae'] if r['hr_mae'] is not None else ''} | {r['bp_loss_type'] or ''} | {r['bp_huber_delta'] if r['bp_huber_delta'] is not None else ''} |"
        )
    OUT_MD.write_text('\n'.join(lines) + '\n')


def main():
    rows = []
    parse_result_jsons(rows)
    parse_robust_log(rows)
    parse_markdown_rows(rows)
    rows = dedup(rows)
    write_outputs(rows)
    print(f'Wrote {OUT_CSV}')
    print(f'Wrote {OUT_MD}')
    print(f'Total rows: {len(rows)}')


if __name__ == '__main__':
    main()
