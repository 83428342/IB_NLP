import os
import wfdb
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from tqdm import tqdm

def download_and_extract_ptbxl(data_dir="../data/ptb-xl"):
    """
    Downloads the PTB-XL dataset from PhysioNet (v1.0.3) if it doesn't exist.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ptb-xl-1.0.3.zip")
    
    # Check if already downloaded and extracted
    if os.path.exists(os.path.join(data_dir, "ptbxl_database.csv")):
        print("PTB-XL database already exists. Skipping download.")
        return data_dir

    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    
    if not os.path.exists(zip_path):
        print(f"Downloading PTB-XL dataset from PhysioNet (approx 500MB)...")
        # Custom hook for tqdm progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
                
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="PTB-XL.zip") as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
            
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract files without the top-level directory wrapper if possible, or just extract all
        zip_ref.extractall(data_dir)
        
    # The zip usually wraps everything in a folder named 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    extracted_folder = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    
    # Move files up one directory for cleaner paths if the wrapper folder exists
    if os.path.exists(extracted_folder):
        import shutil
        for file_name in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, file_name), data_dir)
        os.rmdir(extracted_folder)
        
    print("Extraction complete.")
    return data_dir

import concurrent.futures
import sys

def download_missing_ecg_files(data_dir):
    """
    Reads the ptbxl_database.csv and downloads missing 100Hz .hea and .dat files directly
    from PhysioNet to save time and disk space, using multi-threading for speed.
    """
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        print(f"Downloading {csv_path}...")
        url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
        urllib.request.urlretrieve(url, csv_path)
        
    df = pd.read_csv(csv_path, index_col='ecg_id')
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    
    paths_to_fetch = df['filename_lr'].tolist()
    print(f"Checking and downloading {len(paths_to_fetch)} 100Hz WFDB records using multiple threads...")
    sys.stdout.flush()
    
    def fetch_ecg(path):
        for ext in ['.hea', '.dat']:
            local_file = os.path.join(data_dir, path + ext)
            remote_url = base_url + path + ext
            if not os.path.exists(local_file):
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                try:
                    urllib.request.urlretrieve(remote_url, local_file)
                except Exception:
                    pass
    
    # Use ThreadPoolExecutor for fast parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(fetch_ecg, paths_to_fetch), total=len(paths_to_fetch), desc="Downloading ECGs"))
        
    print("Download phase complete.")
    sys.stdout.flush()

def process_ptbxl_data(data_dir, sampling_rate=100):
    """
    Reads the PTB-XL database CSV, extracts paths, and prepares a merged tabular format 
    pairing the ECG matrix path with auxiliary patient metadata text and the diagnostic label.

    IMPORTANT: We do NOT use the 'report' column because it contains the cardiologist's
    diagnosis (the answer), which would cause label leakage into the text modality.
    Instead, we construct auxiliary text from patient demographics and signal quality metadata,
    which are genuine exogenous context features independent of the label.
    """
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    print(f"Loading metadata from {csv_path}...")
    
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    # 1. Target Label creation (Normal vs Abnormal based on SCP code)
    import ast
    def is_abnormal(scp_dict_str):
        try:
            scp_dict = ast.literal_eval(scp_dict_str)
            if 'NORM' in scp_dict and scp_dict['NORM'] > 0:
                return 0
            return 1
        except:
            return 1
            
    df['Label'] = df['scp_codes'].apply(is_abnormal)
    
    # 2. Build Auxiliary Text from patient demographics and signal quality metadata.
    # These columns describe the PATIENT CONTEXT (exogenous info), NOT the diagnosis.
    def build_auxiliary_text(row):
        # Demographics
        sex = "female" if str(row.get('sex', '')).strip() == '1' else "male"
        age  = row.get('age', 'unknown')
        height = row.get('height', None)
        weight = row.get('weight', None)
        
        bmi_str = ""
        height_str = ""
        weight_str = ""
        try:
            h_val = float(height)
            w_val = float(weight)
            if h_val > 0 and w_val > 0:
                bmi = w_val / ((h_val / 100) ** 2)
                bmi_str = f", BMI {bmi:.1f}"
                height_str = f", {int(h_val)}cm"
                weight_str = f", {int(w_val)}kg"
        except (TypeError, ValueError):
            pass
        
        # Recording context
        site   = str(row.get('site',   'unknown')).strip()
        device = str(row.get('device', 'unknown')).strip()
        
        # Signal quality flags (1.0 = present, 0.0 = absent)
        def flag(col):
            val = row.get(col, 0)
            try:
                return float(val) > 0
            except:
                return False

        quality_issues = []
        if flag('baseline_drift'):     quality_issues.append("baseline drift")
        if flag('static_noise'):       quality_issues.append("static noise")
        if flag('burst_noise'):        quality_issues.append("burst noise")
        if flag('electrodes_problems'): quality_issues.append("electrode problems")
        
        extra_findings = []
        if flag('extra_beats'): extra_findings.append("extra beats detected")
        if flag('pacemaker'):   extra_findings.append("pacemaker present")
        
        quality_str = (
            "Signal quality issues: " + ", ".join(quality_issues) + "."
            if quality_issues else "Signal quality: clean."
        )
        extra_str = (" " + " ".join(extra_findings).capitalize() + ".") if extra_findings else ""
        
        return (
            f"Patient: {age}-year-old {sex}{height_str}{weight_str}{bmi_str}. "
            f"Recording site: {site}. Device: {device}. "
            f"{quality_str}{extra_str}"
        )
    
    df['Article'] = df.apply(build_auxiliary_text, axis=1)
    
    # Quick sanity check: print a sample
    print("\n[Sample auxiliary text]:", df['Article'].iloc[0])
    
    # 3. Path to the actual timeseries .dat/.hea file
    path_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
    df['ECG_Path'] = df[path_col].apply(lambda x: os.path.join(data_dir, x))
    
    # 4. Standardize column names
    output_df = df[['ECG_Path', 'Article', 'Label']].copy()
    output_df = output_df.reset_index(drop=True)
    
    out_csv = os.path.join(data_dir, "PTBXL_MULTI_merged.csv")
    output_df.to_csv(out_csv, index=False)
    
    print(f"Processed {len(output_df)} records.")
    print(f"Label distribution:\n{output_df['Label'].value_counts(normalize=True)}")
    print(f"Data mapping saved to {out_csv}")
    
    return out_csv

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    print("Starting PTB-XL pipeline...")
    download_missing_ecg_files(target_dir)
    process_ptbxl_data(target_dir, sampling_rate=100)
    print("Done!")
