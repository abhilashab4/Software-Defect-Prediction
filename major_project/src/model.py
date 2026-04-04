import torch
import torch.nn as nn

class MultiModalVAE(nn.Module):
    def __init__(self, metrics_dim, vocab_size, embed_dim=64, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.seq_enc = nn.LSTM(embed_dim, 64, batch_first=True, bidirectional=True)
        self.metrics_enc = nn.Sequential(nn.Linear(metrics_dim, 64), nn.ReLU())

        # Probabilistic Bottleneck
        self.fc_fuse = nn.Linear(128 + 64, 64) 
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_metrics, x_seq):
        _, (h_n, _) = self.seq_enc(self.embedding(x_seq))
        h_seq = torch.cat((h_n[0], h_n[1]), dim=-1)
        h_metrics = self.metrics_enc(x_metrics)
        
        h_shared = nn.ReLU()(self.fc_fuse(torch.cat((h_seq, h_metrics), dim=-1)))
        mu, logvar = self.mu(h_shared), self.logvar(h_shared)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, self.classifier(z)