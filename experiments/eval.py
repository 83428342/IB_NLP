import os
import sys
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_loader import get_dataloaders
from models import MultimodalStockPredictor, TimeSeriesOnlyPredictor, TextOnlyPredictor, AblationPredictor
from train import get_model, evaluate

def plot_training_curves(experiments, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot Validation AUROC for all experiments (exclude ablation for clarity)
    plt.subplot(1, 2, 1)
    plot_experiments = [e for e in experiments if e != 'ablation']
    for exp in plot_experiments:
        try:
            history = torch.load(f"{save_dir}/history_{exp}.pt")
            val_auc = [h['auroc'] for h in history['val']]
            plt.plot(val_auc, label=exp, marker='o')
        except FileNotFoundError:
            pass
    plt.title("Validation AUROC across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("AUROC")
    plt.legend()
    plt.grid(True)
    
    # Plot KL Loss of the proposed model
    plt.subplot(1, 2, 2)
    try:
        history = torch.load(f"{save_dir}/history_fusion_vib.pt")
        train_kl = [h['kl_loss'] for h in history['train']]
        plt.plot(train_kl, label="Train KL Loss", color='r')
        plt.title("Information Bottleneck KL Regularization")
        plt.xlabel("Epochs")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid(True)
    except FileNotFoundError:
        pass
    
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=300)
    print("Saved -> loss_curves.png")

def plot_roc_curves(results, save_dir):
    plt.figure(figsize=(8, 8))
    
    for exp, (fpr, tpr, roc_auc) in results.items():
        if exp == 'ablation':
            continue
        plt.plot(fpr, tpr, lw=2, label=f'{exp} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_comparison.png", dpi=300)
    print("Saved -> roc_comparison.png")

def plot_confusion_matrix(y_true, y_pred, exp_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down/Same', 'Up'], 
                yticklabels=['Down/Same', 'Up'])
    plt.title(f'Confusion Matrix - {exp_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{exp_name}.png", dpi=300)
    print(f"Saved -> confusion_matrix_{exp_name}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ptbxl', choices=['ptbxl', 'fnspid'], help="Select data domain")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    if args.dataset == 'ptbxl':
        feature_dim = 12
        window_size = 1000
        bert_model = "emilyalsentzer/Bio_ClinicalBERT"
    else:
        feature_dim = 6
        window_size = 5
        bert_model = "bert-base-uncased"
        
    _, _, test_loader = get_dataloaders(data_dir, dataset_name=args.dataset, batch_size=16, window_size=window_size)
    criterion = nn.CrossEntropyLoss()
    
    experiments = ['ts_only', 'text_only', 'ablation', 'fusion_vib']
    
    roc_results = {}
    save_dir = f"checkpoints_{args.dataset}"
    
    print(f"\n============= Test Evaluation ({args.dataset.upper()}) =============")
    for exp in experiments:
        model_path = f"{save_dir}/best_model_{exp}.pt"
        if not os.path.exists(model_path):
            print(f"Skipping {exp}, checkpoint not found.")
            continue
            
        model = get_model(exp, device, feature_dim, window_size, bert_model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        use_ts = exp in ['ts_only', 'fusion_vib', 'ablation']
        use_text = exp in ['text_only', 'fusion_vib', 'ablation']
        
        metrics, all_probs, all_preds, all_labels = evaluate(model, test_loader, criterion, device, 0.0, use_text, use_ts)
        
        print(f"[{exp.upper()}] Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auroc']:.4f}")
        
        # Calculate ROC data
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_results[exp] = (fpr, tpr, metrics['auroc'])
        
        # Confusion matrix for proposed model
        if exp == 'fusion_vib':
            plot_confusion_matrix(all_labels, all_preds, f"{args.dataset}_{exp}")
            
    if roc_results:
        plot_roc_curves(roc_results, save_dir)
        plot_training_curves(experiments, save_dir)

if __name__ == "__main__":
    main()
