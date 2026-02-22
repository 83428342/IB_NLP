import torch

exps = ['ts_only', 'text_only', 'ablation', 'fusion_vib']
sep = '='*72

print(sep)
print(f"{'Experiment':<16} {'Best AUC':>10} {'Last Acc':>10} {'Last F1':>10} {'Last AUC':>10}")
print(sep)

for exp in exps:
    try:
        h = torch.load(f'checkpoints_ptbxl/history_{exp}.pt')
        val = h['val']
        best_auc = max(v['auroc'] for v in val)
        last = val[-1]
        flag = " <<<" if exp == 'fusion_vib' else ""
        print(f"{exp:<16} {best_auc:>10.4f} {last['acc']:>10.4f} {last['f1']:>10.4f} {last['auroc']:>10.4f}{flag}")
    except Exception as e:
        print(f"{exp:<16}  ERROR: {e}")

print(sep)

# Per-epoch detail for fusion_vib
print("\n[Proposed Model - fusion_vib] Epoch-by-epoch Val metrics:")
print(f"{'Epoch':>6} {'AUC':>8} {'Acc':>8} {'F1':>8} {'KL':>10}")
print('-'*44)
try:
    h = torch.load('checkpoints_ptbxl/history_fusion_vib.pt')
    for i, (t, v) in enumerate(zip(h['train'], h['val'])):
        print(f"{i+1:>6} {v['auroc']:>8.4f} {v['acc']:>8.4f} {v['f1']:>8.4f} {t['kl_loss']:>10.4f}")
except Exception as e:
    print(f"Error: {e}")
