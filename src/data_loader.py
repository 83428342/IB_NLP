import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class FNSPIDMultiDataset(Dataset):
    def __init__(self, data_dir, window_size=5, pretrained_model="bert-base-uncased", split="train"):
        """
        data_dir: dataset directory containing 'FNSPID_MULTI_merged.csv'
        window_size: Number of past days for time-series features
        pretrained_model: BERT model name for tokenizer
        split: 'train', 'val', or 'test'
        """
        self.window_size = window_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
        csv_path = os.path.join(data_dir, "FNSPID_MULTI_merged.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found. Did you run prepare_fnspid.py?")
            
        merged_df = pd.read_csv(csv_path)
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df = merged_df.sort_values(['Stock_symbol', 'Date']).reset_index(drop=True)
        
        self.ts_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # Split chronologically, but we must do it stock-by-stock to maintain balance
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for ticker, group in merged_df.groupby('Stock_symbol'):
            n_total = len(group)
            train_end = int(n_total * 0.8)
            val_end = int(n_total * 0.9)
            
            train_dfs.append(group.iloc[:train_end])
            val_dfs.append(group.iloc[train_end:val_end])
            test_dfs.append(group.iloc[val_end:])
            
        train_combined = pd.concat(train_dfs).reset_index(drop=True)
        val_combined = pd.concat(val_dfs).reset_index(drop=True)
        test_combined = pd.concat(test_dfs).reset_index(drop=True)
        
        # Fit scaler on train data PER TICKER
        self.scalers = {}
        for ticker in train_combined['Stock_symbol'].unique():
            scaler = MinMaxScaler()
            ticker_train_data = train_combined[train_combined['Stock_symbol'] == ticker][self.ts_features]
            scaler.fit(ticker_train_data)
            self.scalers[ticker] = scaler
        
        if split == "train":
            self.df = train_combined.copy()
        elif split == "val":
            self.df = val_combined.copy()
        else:
            self.df = test_combined.copy()
            
        # Apply scaling per ticker
        for ticker in self.df['Stock_symbol'].unique():
            mask = self.df['Stock_symbol'] == ticker
            if ticker in self.scalers:
                self.df.loc[mask, self.ts_features] = self.scalers[ticker].transform(self.df.loc[mask, self.ts_features])
            else:
                # Fallback if a ticker somehow has no train data (rare edge case)
                self.df.loc[mask, self.ts_features] = 0.0
        
        # Pre-compute valid indices. We cannot slide a window across different stocks!
        self.valid_indices = []
        for ticker, group in self.df.groupby('Stock_symbol'):
            # The indices of this group in the main dataframe
            group_indices = group.index.values
            if len(group_indices) > self.window_size:
                # We can form valid windows form group_indices[0] to group_indices[-window_size-1]
                for i in range(len(group_indices) - self.window_size):
                    self.valid_indices.append(group_indices[i])
                    
        self.valid_indices = np.array(self.valid_indices)
        
        self.timestamps = self.df['Date'].values
        self.labels = self.df['Label'].values
        self.ts_data = self.df[self.ts_features].values
        
        self.df['Article'] = self.df['Article'].fillna('').astype(str)
        self.combined_news = self.df['Article'].tolist()
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Actual starting index in the dataframe
        start_idx = self.valid_indices[idx]
        window_end_idx = start_idx + self.window_size
        current_idx = window_end_idx - 1 # The "current day" we are making predictions on
        
        x_ts = self.ts_data[start_idx:window_end_idx]
        text = self.combined_news[current_idx]
        label = self.labels[current_idx]
        
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        
        return {
            'x_ts': torch.tensor(x_ts, dtype=torch.float32),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

import wfdb

class PTBXLDataset(Dataset):
    def __init__(self, data_dir, pretrained_model="emilyalsentzer/Bio_ClinicalBERT", split="train"):
        """
        data_dir: dataset directory containing 'PTBXL_MULTI_merged.csv' and actual waveform files
        """
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
        csv_path = os.path.join(data_dir, "PTBXL_MULTI_merged.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found. Did you run prepare_ptbxl.py?")
            
        df = pd.read_csv(csv_path)
        
        # Standard Train/Val/Test Split for PTB-XL (often 80/10/10)
        n_total = len(df)
        train_end = int(n_total * 0.8)
        val_end = int(n_total * 0.9)
        
        if split == "train":
            self.df = df.iloc[:train_end].copy().reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        else:
            self.df = df.iloc[val_end:].copy().reset_index(drop=True)
            
        self.paths = self.df['ECG_Path'].values
        self.texts = self.df['Article'].values
        self.labels = self.df['Label'].values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Load WFDB record. 
        # wfdb.rdsamp returns (signal_matrix, metadata_dict)
        # Expected shape for 100Hz PTB-XL is usually (1000, 12) i.e. 10 seconds * 100Hz
        try:
            signals, fields = wfdb.rdsamp(path)
        except Exception:
            # Fallback zero tensor if file is missing/corrupted
            signals = np.zeros((1000, 12), dtype=np.float32)
            
        # Ensure exact sequence length (padding or truncation)
        expected_len = 1000
        if signals.shape[0] < expected_len:
            pad_width = ((0, expected_len - signals.shape[0]), (0, 0))
            x_ts = np.pad(signals, pad_width, mode='constant')
        else:
            x_ts = signals[:expected_len, :]
            
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128, # shorter length for clinical notes
            return_tensors='pt'
        )
        
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        
        return {
            'x_ts': torch.tensor(x_ts, dtype=torch.float32), # (1000, 12)
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(data_dir, dataset_name='ptbxl', batch_size=32, window_size=5):
    if dataset_name == 'fnspid':
        train_dataset = FNSPIDMultiDataset(data_dir, window_size=window_size, split="train")
        val_dataset = FNSPIDMultiDataset(data_dir, window_size=window_size, split="val")
        test_dataset = FNSPIDMultiDataset(data_dir, window_size=window_size, split="test")
    elif dataset_name == 'ptbxl':
        train_dataset = PTBXLDataset(data_dir, split="train")
        val_dataset = PTBXLDataset(data_dir, split="val")
        test_dataset = PTBXLDataset(data_dir, split="test")
    else:
        raise ValueError(f"Unknown dataset mode: {dataset_name}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    import sys
    d_name = sys.argv[1] if len(sys.argv) > 1 else 'ptbxl'
    train_loader, val_loader, test_loader = get_dataloaders(dataset_dir, dataset_name=d_name, batch_size=4)
    
    for batch in train_loader:
        print(f"[{d_name.upper()}] Dataset size:", len(train_loader.dataset))
        print("x_ts shape:", batch['x_ts'].shape)
        print("input_ids shape:", batch['input_ids'].shape)
        print("attention_mask shape:", batch['attention_mask'].shape)
        print("label shape:", batch['label'].shape)
        print("Label values:", batch['label'])
        break
