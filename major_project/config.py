import torch

# Project Configuration
PROJECT_NAME = "camel"  # Change to "camel", "jedit", etc.
TRAIN_VERSIONS = ["1.4"]
TEST_VERSION = "1.6"

# Model Hyperparameters
METRICS_DIM = 20
VOCAB_SIZE = 5000
EMBED_DIM = 64
LATENT_DIM = 32

# Training Hyperparameters
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 200
KL_COEFF = 0.05

# Environment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f"models/{PROJECT_NAME}_vae_best_64.pth"
VOCAB_SAVE_PATH = f"models/{PROJECT_NAME}_vocab_64.pkl"