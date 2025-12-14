import numpy as np
import time
from typing import Tuple, Dict
from base import BaseQuantizer, ScaleType

class NaiveQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__("Naive (BitNet)")

    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Dict]:
        start_time = time.time()
        
        # TODO: Implement AbsMean calculation
        # 1. Calculate alpha = mean(|W|)
        # 2. Q = round(W / alpha)
        # 3. Clip to {-1, 1}
        
        # --- STUB ---
        alpha = 1.0
        Q = np.zeros_like(W) 
        # ------------
        
        elapsed = time.time() - start_time
        return Q, alpha, {"time": elapsed}

class ALSQuantizer(BaseQuantizer):
    def __init__(self, max_iter=10, row_wise=False, init_svd=False):
        name = f"ALS{' (Row-wise)' if row_wise else ''}{' + SVD' if init_svd else ''}"
        super().__init__(name)
        self.max_iter = max_iter
        self.row_wise = row_wise
        self.init_svd = init_svd

    def quantize(self, W: np.ndarray) -> Tuple[np.ndarray, ScaleType, Dict]:
        start_time = time.time()
        history = []
        
        # TODO: Implement ALS
        # 1. Init alpha and Q (Random or SVD if self.init_svd)
        # 2. Loop self.max_iter:
        #    a. Fix Q, solve alpha (Least Squares)
        #    b. Fix alpha, solve Q (Thresholding)
        #    c. Calc error, append to history
        
        # --- STUB ---
        N, M = W.shape
        alpha = np.ones((N, 1)) if self.row_wise else 1.0
        Q = np.sign(W)
        history = [0.9, 0.5, 0.1] # Fake convergence как заглушка
        # ------------
        
        elapsed = time.time() - start_time
        return Q, alpha, {"loss_history": history, "time": elapsed}