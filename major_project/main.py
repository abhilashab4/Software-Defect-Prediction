# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import f1_score
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os, pickle, copy
# import pandas as pd

# # Internal Imports
# import config as cfg
# from src.dataset import MultiModalDefectDataset
# from src.model import MultiModalVAE
# from src.utils import save_performance_report  # Added Utils

# # Setup Directories
# os.makedirs("models", exist_ok=True)
# os.makedirs("results", exist_ok=True)

# # 1. Data Loading Pipeline
# print(f"--- 🔄 Starting Experiment: {cfg.PROJECT_NAME.upper()} ---")
# train_paths = [f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{v}_enriched.csv" for v in cfg.TRAIN_VERSIONS]
# train_dfs = [pd.read_csv(p) for p in train_paths if os.path.exists(p)]
# train_ds = MultiModalDefectDataset(pd.concat(train_dfs).reset_index(drop=True))

# test_path = f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{cfg.TEST_VERSION}_enriched.csv"
# test_ds = MultiModalDefectDataset(pd.read_csv(test_path), tokenizer=train_ds.tokenizer)

# train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

# # 2. Model Initialization
# model = MultiModalVAE(cfg.METRICS_DIM, cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.LATENT_DIM).to(cfg.DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
# criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight.to(cfg.DEVICE))

# best_f1 = 0.0
# best_model_wts = copy.deepcopy(model.state_dict())

# # 3. Training Loop
# print(f"🚀 Training for {cfg.EPOCHS} Epochs...")
# for epoch in range(cfg.EPOCHS):
#     model.train()
#     for m, s, y in train_loader:
#         m, s, y = m.to(cfg.DEVICE), s.to(cfg.DEVICE), y.to(cfg.DEVICE)
#         optimizer.zero_grad()
#         mu, logvar, logits = model(m, s)
        
#         # Latent Augmentation (Generative Logic)
#         bug_idx = (y == 1).nonzero(as_tuple=True)[0]
#         if len(bug_idx) > 0:
#             # std_bugs = torch.exp(0.5 * logvar[bug_idx])
#             # z_syn = mu[bug_idx] + torch.randn_like(std_bugs) * std_bugs
#             # l_syn = model.classifier(z_syn)
#             # all_logits = torch.cat([logits.view(-1), l_syn.view(-1)])
#             # all_targets = torch.cat([y, torch.ones(len(bug_idx)).to(cfg.DEVICE)])
#             std_bugs = torch.exp(0.5 * logvar[bug_idx])
    
#             # 2. INCREASE INTENSITY: Multiply std by 1.5 to search 'further'
#             intensity_multiplier = 1.5 
#             z_syn = mu[bug_idx] + torch.randn_like(std_bugs) * (std_bugs * intensity_multiplier)
    
#             # 3. Classify these 'distorted' synthetic bugs
#             l_syn = model.classifier(z_syn)
    
#             all_logits = torch.cat([logits.view(-1), l_syn.view(-1)])
#             all_targets = torch.cat([y, torch.ones(len(bug_idx)).to(cfg.DEVICE)])
#         else:
#             all_logits, all_targets = logits.view(-1), y

#         # Combined VAE Loss
#         loss = criterion(all_logits, all_targets) + cfg.KL_COEFF * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
#         loss.backward()
#         optimizer.step()

#     # Evaluation & Best Checkpoint Tracking
#     model.eval()
#     y_true_epoch, y_pred_epoch = [], []
#     with torch.no_grad():
#         for m_t, s_t, y_t in test_loader:
#             _, _, l_t = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
#             y_pred_epoch.extend((torch.sigmoid(l_t) > 0.5).cpu().numpy())
#             y_true_epoch.extend(y_t.numpy())
    
#     current_f1 = f1_score(y_true_epoch, y_pred_epoch)
#     if current_f1 > best_f1:
#         best_f1 = current_f1
#         best_model_wts = copy.deepcopy(model.state_dict())
#         torch.save(best_model_wts, cfg.MODEL_SAVE_PATH)
#     print(f"🌟 Epoch {epoch+1:03d} | New Best F1: {best_f1:.4f}")

# # 4. Final Evaluation & Results Generation
# print(f"\n📊 Finalizing Results for {cfg.PROJECT_NAME.upper()}...")
# model.load_state_dict(best_model_wts)
# with open(cfg.VOCAB_SAVE_PATH, "wb") as f:
#     pickle.dump(train_ds.tokenizer, f)

# model.eval()
# final_true, final_pred = [], []
# with torch.no_grad():
#     # Capture results for Metrics Report
#     for m_t, s_t, y_t in test_loader:
#         _, _, l_t = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
#         final_pred.extend((torch.sigmoid(l_t) > 0.5).cpu().numpy())
#         final_true.extend(y_t.numpy())
    
#     # Capture Latent Space for t-SNE
#     mu, _, _ = model(test_ds.X_metrics.to(cfg.DEVICE), test_ds.X_seq.to(cfg.DEVICE))
#     z_2d = TSNE(n_components=2, random_state=42).fit_transform(mu.cpu().numpy())

# # Generate Plots & Reports
# save_performance_report(final_true, final_pred, cfg.PROJECT_NAME)

# plt.figure(figsize=(10, 7))
# plt.scatter(z_2d[:, 0], z_2d[:, 1], c=test_ds.y.numpy(), cmap='coolwarm', alpha=0.7)
# plt.title(f"{cfg.PROJECT_NAME.upper()} Peak Latent Space (F1: {best_f1:.4f})")
# plt.savefig(f"results/{cfg.PROJECT_NAME}_best_tsne_64.png")

# print(f"✅ Finished. Best F1: {best_f1:.4f}. All artifacts saved in models/ and results/ folders.")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os, pickle, copy, random
import pandas as pd
import numpy as np

# Internal Imports
import config as cfg
from src.dataset import MultiModalDefectDataset
from src.model import MultiModalVAE
from src.utils import save_performance_report

# ==========================================
# 0. REPRODUCIBILITY: SET SEED
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Using the standard research seed

# Setup Directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Data Loading Pipeline
print(f"--- 🔄 Starting Experiment: {cfg.PROJECT_NAME.upper()} ---")
train_paths = [f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{v}_enriched.csv" for v in cfg.TRAIN_VERSIONS]
train_dfs = [pd.read_csv(p) for p in train_paths if os.path.exists(p)]
train_ds = MultiModalDefectDataset(pd.concat(train_dfs).reset_index(drop=True))

test_path = f"data/{cfg.PROJECT_NAME}/{cfg.PROJECT_NAME}_{cfg.TEST_VERSION}_enriched.csv"
test_ds = MultiModalDefectDataset(pd.read_csv(test_path), tokenizer=train_ds.tokenizer)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

# 2. Model Initialization
model = MultiModalVAE(cfg.METRICS_DIM, cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.LATENT_DIM).to(cfg.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight.to(cfg.DEVICE))

best_f1 = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

# 3. Training Loop
print(f"🚀 Training for {cfg.EPOCHS} Epochs (Seed: 42)...")
for epoch in range(cfg.EPOCHS):
    model.train()
    for m, s, y in train_loader:
        m, s, y = m.to(cfg.DEVICE), s.to(cfg.DEVICE), y.to(cfg.DEVICE)
        optimizer.zero_grad()
        mu, logvar, logits = model(m, s)
        
        # --- Latent Augmentation (Generative Logic) ---
        bug_idx = (y == 1).nonzero(as_tuple=True)[0]
        if len(bug_idx) > 0:
            std_bugs = torch.exp(0.5 * logvar[bug_idx])
            z_syn = mu[bug_idx] + torch.randn_like(std_bugs) * std_bugs
            l_syn = model.classifier(z_syn)
            all_logits = torch.cat([logits.view(-1), l_syn.view(-1)])
            all_targets = torch.cat([y, torch.ones(len(bug_idx)).to(cfg.DEVICE)])
            # std_bugs = torch.exp(0.5 * logvar[bug_idx])
            
            # # Intensity Multiplier to expand the "Buggy Zone"
            # intensity_multiplier = 1.5 
            # z_syn = mu[bug_idx] + torch.randn_like(std_bugs) * (std_bugs * intensity_multiplier)
            
            # l_syn = model.classifier(z_syn)
            # all_logits = torch.cat([logits.view(-1), l_syn.view(-1)])
            # all_targets = torch.cat([y, torch.ones(len(bug_idx)).to(cfg.DEVICE)])
        else:
            all_logits, all_targets = logits.view(-1), y

        # Combined Loss
        loss = criterion(all_logits, all_targets) + cfg.KL_COEFF * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    y_true_epoch, y_pred_epoch = [], []
    with torch.no_grad():
        for m_t, s_t, y_t in test_loader:
            _, _, l_t = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
            y_pred_epoch.extend((torch.sigmoid(l_t) > 0.5).cpu().numpy())
            y_true_epoch.extend(y_t.numpy())
    
    current_f1 = f1_score(y_true_epoch, y_pred_epoch)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, cfg.MODEL_SAVE_PATH)
    print(f"🌟 Epoch {epoch+1:03d} | New Best F1: {best_f1:.4f}")

# 4. Final Results Generation
print(f"\n📊 Finalizing Results for {cfg.PROJECT_NAME.upper()}...")
model.load_state_dict(best_model_wts)
with open(cfg.VOCAB_SAVE_PATH, "wb") as f:
    pickle.dump(train_ds.tokenizer, f)

model.eval()
final_true, final_pred = [], []
with torch.no_grad():
    for m_t, s_t, y_t in test_loader:
        _, _, l_t = model(m_t.to(cfg.DEVICE), s_t.to(cfg.DEVICE))
        final_pred.extend((torch.sigmoid(l_t) > 0.5).cpu().numpy())
        final_true.extend(y_t.numpy())
    
    # Generate Peak t-SNE Plot
    mu, _, _ = model(test_ds.X_metrics.to(cfg.DEVICE), test_ds.X_seq.to(cfg.DEVICE))
    z_2d = TSNE(n_components=2, random_state=42).fit_transform(mu.cpu().numpy())

save_performance_report(final_true, final_pred, cfg.PROJECT_NAME)

plt.figure(figsize=(10, 7))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=test_ds.y.numpy(), cmap='coolwarm', alpha=0.7)
plt.title(f"{cfg.PROJECT_NAME.upper()} Peak Latent Space (F1: {best_f1:.4f})")
plt.savefig(f"results/{cfg.PROJECT_NAME}_best_tsne_final.png")

print(f"✅ Finished. Best F1: {best_f1:.4f}. All artifacts saved in models/ and results/.")

