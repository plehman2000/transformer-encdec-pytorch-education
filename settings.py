import logging
import os
import random

import numpy as np
import torch

os.environ["WANDB_API_KEY"] = "4a67020c6e449cbdc59e85ce39017587388bca6d"

logging.basicConfig(level=logging.DEBUG)

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
