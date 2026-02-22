import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_loader import get_dataloaders
from models import MultimodalStockPredictor, TimeSeriesOnlyPredictor, TextOnlyPredictor, AblationPredictor

def get_model(experiment, device, feature_dim, window_size, bert_model_name):
    if experiment == "ts_only":
        model = TimeSeriesOnlyPredictor(ts_feature_dim=feature_dim, window_size=window_size)
    elif experiment == "text_only":
        model = TextOnlyPredictor(bert_model_name=bert_model_name)
    elif experiment == "fusion_vib":
        model = MultimodalStockPredictor(ts_feature_dim=feature_dim, window_size=window_size, bert_model_name=bert_model_name)
    elif experiment == "ablation":
        model = AblationPredictor(ts_feature_dim=feature_dim, window_size=window_size, bert_model_name=bert_model_name)
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")
        
    return model.to(device)

def calculate_metrics(y_true, y_pred, y_prob):
    # Safety check for AUROC when only one class is present in batch
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5
        
    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'auroc': auroc
    }

def train_one_epoch(model, dataloader, optimizer, criterion, device, beta, use_text, use_ts):
    model.train()
    total_loss, total_task_loss, total_kl_loss = 0, 0, 0
    
    all_preds, all_probs, all_labels = [], [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        labels = batch['label'].to(device)
        
        args = []
        if use_ts:
            args.append(batch['x_ts'].to(device))
        if use_text:
            args.extend([batch['input_ids'].to(device), batch['attention_mask'].to(device)])
            
        logits, kl_loss = model(*args)
        
        task_loss = criterion(logits, labels)
        loss = task_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
            
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return {
        'loss': total_loss / len(dataloader),
        'task_loss': total_task_loss / len(dataloader),
        'kl_loss': total_kl_loss / len(dataloader),
        **metrics
    }

def evaluate(model, dataloader, criterion, device, beta, use_text, use_ts):
    model.eval()
    total_loss, total_task_loss, total_kl_loss = 0, 0, 0
    
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label'].to(device)
            
            args = []
            if use_ts:
                args.append(batch['x_ts'].to(device))
            if use_text:
                args.extend([batch['input_ids'].to(device), batch['attention_mask'].to(device)])
                
            logits, kl_loss = model(*args)
            
            task_loss = criterion(logits, labels)
            loss = task_loss + beta * kl_loss
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return {
        'loss': total_loss / len(dataloader),
        'task_loss': total_task_loss / len(dataloader),
        'kl_loss': total_kl_loss / len(dataloader),
        **metrics
    }, all_probs, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ptbxl', choices=['ptbxl', 'fnspid'], help="Select data domain")
    parser.add_argument('--exp', type=str, default='fusion_vib', choices=['ts_only', 'text_only', 'fusion_vib', 'ablation'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.001, help="KL Divergence weight")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset specific configurations
    if args.dataset == 'ptbxl':
        feature_dim = 12
        window_size = 1000
        bert_model = "emilyalsentzer/Bio_ClinicalBERT"
    else:
        # fnspid
        feature_dim = 6
        window_size = 5
        bert_model = "bert-base-uncased"
        
    print(f"Dataset: {args.dataset.upper()} | Features: {feature_dim} | Window: {window_size}")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, dataset_name=args.dataset, batch_size=args.batch_size, window_size=window_size)
    
    model = get_model(args.exp, device, feature_dim, window_size, bert_model)
    criterion = nn.CrossEntropyLoss()
    
    # Do not optimize frozen BERT
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    use_ts = args.exp in ['ts_only', 'fusion_vib', 'ablation']
    use_text = args.exp in ['text_only', 'fusion_vib', 'ablation']
    
    best_val_auc = 0
    save_dir = f"checkpoints_{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} - Experiment: {args.exp}")
        
        # KL Annealing: linearly increase beta from 0 to target over the first half of training
        # warmup_epochs is tied to args.epochs so beta always reaches its full value mid-training.
        warmup_epochs = max(args.epochs // 2, 5)
        current_beta = args.beta * min(1.0, (epoch + 1) / warmup_epochs)
        if current_beta > 0:
            print(f"Current Beta (KL Weight): {current_beta:.6f}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, current_beta, use_text, use_ts)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, current_beta, use_text, use_ts)
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"Train | Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, AUC: {train_metrics['auroc']:.4f}")
        print(f"Val   | Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auroc']:.4f}")
        
        if val_metrics['auroc'] > best_val_auc:
            best_val_auc = val_metrics['auroc']
            torch.save(model.state_dict(), f"{save_dir}/best_model_{args.exp}.pt")
            print(">>> Saved best model!")
            
    # Save history for plotting
    torch.save(history, f"{save_dir}/history_{args.exp}.pt")
    print("\nTraining Completed.")

if __name__ == "__main__":
    main()
