import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from helper import evaluate, set_seed

set_seed(42)

class YelpDataset(Dataset):
    def __init__(self, texts, controls, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.controls = torch.tensor(controls, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "controls": self.controls[idx],
            "labels": self.labels[idx]
        }

class RoBERTaWithControls(nn.Module):
    def __init__(self, model_name="roberta-base", num_controls=2, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            add_pooling_layer=False
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        combined_dim = self.encoder.config.hidden_size + num_controls
        self.regressor = nn.Linear(combined_dim, 1)

    def forward(self, input_ids, attention_mask, controls):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate text embedding with control features
        combined = torch.cat([cls_embedding, controls], dim=1)
        return self.regressor(combined).squeeze(-1)

print("Loading dataset and preparing 1,000 sample benchmark...")
df = pd.read_parquet("dataset/yelp_train.parquet")
df["review_year"] = df["date"].dt.year
df["business_review_count"] = df.groupby("business_id")["review_id"].transform("count")
df["log_useful"] = np.log1p(df["useful"])

# Take 1,000 samples for timing (800 train, 200 val)
sample_df = df.sample(n=1000, random_state=42).reset_index(drop=True)
train_df = sample_df.iloc[:800]
val_df = sample_df.iloc[800:]

# Line 247: Standardize controls using training set statistics only
control_cols = ["review_year", "business_review_count"]
scaler = StandardScaler()
train_controls = scaler.fit_transform(train_df[control_cols])
val_controls = scaler.transform(val_df[control_cols])

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = YelpDataset(train_df["text"].tolist(), train_controls, train_df["log_useful"].tolist(), tokenizer)
val_dataset = YelpDataset(val_df["text"].tolist(), val_controls, val_df["log_useful"].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def run_test(freeze):
    mode = "Frozen RoBERTa" if freeze else "Fine Tuned RoBERTa"
    model = RoBERTaWithControls(model_name=model_name, num_controls=2, freeze_encoder=freeze).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)
    criterion = nn.MSELoss()
    
    # Time 1 training epoch
    model.train()
    start_time = time.perf_counter()
    for batch in train_loader:
        optimizer.zero_grad()
        preds = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["controls"].to(device)
        )
        loss = criterion(preds, batch["labels"].to(device))
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start_time
    
    # Evaluate on validation slice
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["controls"].to(device)
            )
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
            
    metrics = evaluate(all_labels, all_preds)
    print(f"\n{mode} Results:")
    print(f"Time (800 train samples, 1 epoch): {elapsed:.2f} seconds")
    print(f"RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | Spearman_rho: {metrics['Spearman_rho']:.4f}")

run_test(freeze=True)
run_test(freeze=False)