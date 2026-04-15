import torch
import pandas as pd
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MultiModalDefectDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_len=128):
        # 1. Metrics (Dynamic extraction between first and last 3 columns)
        metrics = df.iloc[:, 1:-3].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Z-score Normalization for VAE stability
        normalized = (metrics - metrics.mean()) / (metrics.std() + 1e-8)
        self.X_metrics = torch.tensor(normalized.values, dtype=torch.float32)
        
        # 2. Sequence (AST + Tokens)
        combined_text = df['ast_seq'].astype(str) + " " + df['code_tokens'].astype(str)
        
        if tokenizer is None:
            self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(combined_text)
        else:
            self.tokenizer = tokenizer 
            
        seqs = self.tokenizer.texts_to_sequences(combined_text)
        self.X_seq = torch.tensor(pad_sequences(seqs, maxlen=max_len), dtype=torch.long)
        
        # 3. Binary Label (bug > 0)
        self.y = torch.tensor((df.iloc[:, -3] > 0).astype(int).values, dtype=torch.float32)
        
        # 4. Class Weighting
        num_clean = (self.y == 0).sum().item()
        num_bug = (self.y == 1).sum().item()
        self.pos_weight = torch.tensor([num_clean / max(num_bug, 1)], dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_metrics[idx], self.X_seq[idx], self.y[idx]

# import torch
# import pandas as pd
# from torch.utils.data import Dataset
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# class MultiModalDefectDataset(Dataset):
#     def __init__(
#         self,
#         df,
#         tokenizer=None,
#         max_len=128,
#         mean=None,
#         std=None,
#         fit_tokenizer=False
#     ):
#         # -----------------------------
#         # 1. METRICS (NO LEAKAGE)
#         # -----------------------------
#         metrics = df.iloc[:, 1:-3].apply(pd.to_numeric, errors='coerce').fillna(0)

#         if mean is None or std is None:
#             self.mean = metrics.mean()
#             self.std = metrics.std()
#         else:
#             self.mean = mean
#             self.std = std

#         normalized = (metrics - self.mean) / (self.std + 1e-8)
#         self.X_metrics = torch.tensor(normalized.values, dtype=torch.float32)

#         # -----------------------------
#         # 2. SEQUENCE (NO TOKENIZER LEAKAGE)
#         # -----------------------------
#         combined_text = df['ast_seq'].astype(str) + " " + df['code_tokens'].astype(str)

#         if tokenizer is None:
#             self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#         else:
#             self.tokenizer = tokenizer

#         if fit_tokenizer:
#             self.tokenizer.fit_on_texts(combined_text)

#         seqs = self.tokenizer.texts_to_sequences(combined_text)

#         self.X_seq = torch.tensor(
#             pad_sequences(
#                 seqs,
#                 maxlen=max_len,
#                 padding='post',
#                 truncating='post'
#             ),
#             dtype=torch.long
#         )

#         # -----------------------------
#         # 3. LABEL
#         # -----------------------------
#         self.y = torch.tensor(
#             (df.iloc[:, -3] > 0).astype(int).values,
#             dtype=torch.float32
#         )

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X_metrics[idx], self.X_seq[idx], self.y[idx]