import numpy as np
from data import get_synthetic_matrix
from solvers import NaiveQuantizer, ALSQuantizer
from metrics import frobenius_error

def run_experiment():
    # 1. Setup Data
    print("Generating data...")
    W = get_synthetic_matrix('gaussian', (1024, 1024))
    
    # 2. Setup Solvers
    solvers = [
        NaiveQuantizer(),
        ALSQuantizer(max_iter=5, row_wise=False),
        ALSQuantizer(max_iter=5, row_wise=True)
    ]
    
    # 3. Run Loop
    results = {}
    print(f"{'Solver':<20} | {'Error':<10} | {'Time (s)':<10}")
    print("-" * 50)
    
    for solver in solvers:
        # The Interface Contract in action:
        Q, alpha, stats = solver.quantize(W)
        
        # The Metrics Contract in action:
        error = frobenius_error(W, Q, alpha)
        
        print(f"{solver.name:<20} | {error:.4f}     | {stats['time']:.4f}")
        results[solver.name] = error

if __name__ == "__main__":
    run_experiment()