import torch
import torch.nn as nn
from transformers import AutoModel
import math

class VIBLayer(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(VIBLayer, self).__init__()
        self.z_dim = z_dim
        self.fc_mu = nn.Linear(input_dim, z_dim)
        self.fc_logvar = nn.Linear(input_dim, z_dim)
        
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Return z and KL divergence items
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        return z, kl_loss

class TimeSeriesTransformerVIB(nn.Module):
    def __init__(self, feature_dim, window_size, d_model=64, n_heads=4, n_layers=2, z_dim=64):
        super(TimeSeriesTransformerVIB, self).__init__()
        self.d_model = d_model
        
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, window_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.vib = VIBLayer(input_dim=d_model, z_dim=z_dim)
        
    def forward(self, x):
        # x shape: (batch_size, window_size, feature_dim)
        x = self.embedding(x) * math.sqrt(self.d_model) + self.pos_encoder
        
        # out shape: (batch_size, window_size, d_model)
        out = self.transformer_encoder(x)
        
        # Mean pooling over all timesteps — captures full sequence context
        # (vs last-token only, which loses most of the 1000-step ECG signal)
        out_pooled = out.mean(dim=1)
        
        z_ts, kl_loss = self.vib(out_pooled)
        return z_ts, kl_loss

class TextBERTEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", freeze=True, proj_dim=32):
        super(TextBERTEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.out_dim = proj_dim 
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Unfreeze the last encoder layer and the pooler for domain adaptation
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
                
        # Project 768 to 32 to match Time-Series latent dimension and ensure gradients don't wash out TS
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation for downstream tasks
        pooler_output = outputs.pooler_output
        return self.projection(pooler_output)

class MultimodalStockPredictor(nn.Module):
    """Proposed model: VIB on TS encoder → clean Z_ts, then append Z_text as auxiliary context.
    
    Design intent:
      X_ts → [Transformer + VIB] → Z_ts   (minimal sufficient TS representation)
                                       ↓ concat
                    X_text → [BERT] → Z_text  (auxiliary demographic/context signal)
                                       ↓
                                 [Classifier] → Y
    
    The VIB removes noise/irrelevant variance from the ECG signal.
    Text embeddings are not bottlenecked — they act as side information.
    """
    def __init__(self, 
                 ts_feature_dim, 
                 window_size, 
                 z_dim=64,          # matches AblationPredictor d_model=64 for fair comparison
                 bert_model_name="bert-base-uncased",
                 freeze_bert=True,
                 dropout_rate=0.3):
        super(MultimodalStockPredictor, self).__init__()
        
        # TS encoder with VIB — mean pooling over full sequence
        self.ts_encoder = TimeSeriesTransformerVIB(feature_dim=ts_feature_dim, 
                                                   window_size=window_size, 
                                                   z_dim=z_dim)
        # Text encoder (auxiliary)
        self.text_encoder = TextBERTEncoder(pretrained_model=bert_model_name, 
                                            freeze=freeze_bert)
        
        # Fusion Classifier: [Z_ts || Z_text] → Y
        bert_dim = self.text_encoder.out_dim
        fusion_dim = z_dim + bert_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
        
    def forward(self, x_ts, input_ids, attention_mask):
        z_ts, kl_loss = self.ts_encoder(x_ts)   # VIB compresses TS
        z_text = self.text_encoder(input_ids, attention_mask)  # auxiliary context
        z_fused = torch.cat([z_ts, z_text], dim=1)
        logits = self.classifier(z_fused)
        return logits, kl_loss

class TimeSeriesOnlyPredictor(nn.Module):
    def __init__(self, ts_feature_dim, window_size, z_dim=64, dropout_rate=0.3):
        super(TimeSeriesOnlyPredictor, self).__init__()
        self.ts_encoder = TimeSeriesTransformerVIB(feature_dim=ts_feature_dim, window_size=window_size, z_dim=z_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x_ts):
        z_ts, kl_loss = self.ts_encoder(x_ts)
        logits = self.classifier(z_ts)
        return logits, kl_loss

class TextOnlyPredictor(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", freeze_bert=True, dropout_rate=0.3):
        super(TextOnlyPredictor, self).__init__()
        self.text_encoder = TextBERTEncoder(pretrained_model=bert_model_name, freeze=freeze_bert)
        bert_dim = self.text_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(bert_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, input_ids, attention_mask):
        z_text = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(z_text)
        return logits, 0.0 # Return 0.0 for kl_loss to keep API consistent

class AblationPredictor(nn.Module):
    """Ablation: same architecture as MultimodalStockPredictor but WITHOUT VIB.
    
    Identical to fusion_vib except:
      - No VIB layer (no KL loss, no reparameterisation)
      - d_model=64 (same latent capacity as z_dim=64 in fusion_vib)
      - dropout=0.2 in Transformer (matches fusion_vib)
      - mean pooling (matches fusion_vib)
    This ensures ablation vs fusion_vib isolates the sole effect of VIB.
    """
    def __init__(self, ts_feature_dim=6, window_size=5, d_model=64,
                 bert_model_name="bert-base-uncased", freeze_bert=True, dropout_rate=0.3):
        super(AblationPredictor, self).__init__()
        # TS encoder WITHOUT VIB — same structure as TimeSeriesTransformerVIB otherwise
        self.embedding = nn.Linear(ts_feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True, dropout=0.2)  # dropout=0.2 matches fusion_vib
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Text Encoder (same as fusion_vib)
        self.text_encoder = TextBERTEncoder(pretrained_model=bert_model_name, freeze=freeze_bert)
        bert_dim = self.text_encoder.out_dim
        
        # Fusion Classifier (identical to fusion_vib)
        fusion_dim = d_model + bert_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )

    def forward(self, x_ts, input_ids, attention_mask):
        x = self.embedding(x_ts) * math.sqrt(self.embedding.out_features) + self.pos_encoder
        out = self.transformer_encoder(x)
        z_ts = out.mean(dim=1)   # mean pooling — identical to fusion_vib
        z_text = self.text_encoder(input_ids, attention_mask)
        z_fused = torch.cat([z_ts, z_text], dim=1)
        logits = self.classifier(z_fused)
        return logits, 0.0  # No KL loss (ablation of VIB)
