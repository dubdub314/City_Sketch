

import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CSVLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.header_written = False
    def log(self, **kwargs):
        if not self.header_written:
            self.f.write(",".join(kwargs.keys()) + "\n")
            self.header_written = True
        self.f.write(",".join(str(kwargs[k]) for k in kwargs.keys()) + "\n")
        self.f.flush()
    def close(self):
        self.f.close()
