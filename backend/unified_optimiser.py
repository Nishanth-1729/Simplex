import cvxpy as cp
import numpy as np
from typing import Dict, Optional
from enum import Enum
from dataclasses import dataclass
import time

class UtilityType(Enum):
    # Types of utility functions
    LOG = "log"
    SQRT = "sqrt"
    LINEAR = "linear"
    ALPHA_FAIR = "alpha_fair"

@dataclass
class OptimizationResult:
    # Stores the final optimization result
    status: str
    allocation: np.ndarray
    objective_value: float
    fairness_score: float
    efficiency_score: float
    latency_score: float
    robustness_score: float
    metrics: Dict
    solve_time: float

class UnifiedOptimizer:
    # Multi-objective bandwidth optimizer
    
    def __init__(self, n_users: int, total_capacity: float):
        self.n_users = n_users
        self.total_capacity = total_capacity
    
    def log_utility(self, allocations: cp.Variable) -> cp.Expression:
        # U(x) = sum(log(x_i))
        return cp.sum(cp.log(allocations + 1e-6))
    
    def sqrt_utility(self, allocations: cp.Variable) -> cp.Expression:
        # U(x) = sum(sqrt(x_i))
        return cp.sum(cp.sqrt(allocations))
    
    def linear_utility(self, allocations: cp.Variable) -> cp.Expression:
        # U(x) = sum(x_i)
        return cp.sum(allocations)
    
    def alpha_fair_utility(self, allocations: cp.Variable, alpha: float = 1.0) -> cp.Expression:
        # Alpha-fair utility function
        if abs(alpha - 1.0) < 1e-6:
            return self.log_utility(allocations)
        return cp.sum(cp.power(allocations + 1e-6, 1 - alpha)) / (1 - alpha)
    
    def calculate_jains_index(self, allocations: np.ndarray) -> float:
        # Jain’s fairness index
        n = len(allocations)
        sum_x = np.sum(allocations)
        sum_x2 = np.sum(allocations ** 2)
        if sum_x2 == 0:
            return 0.0
        return (sum_x ** 2) / (n * sum_x2)
    
    def calculate_gini_coefficient(self, allocations: np.ndarray) -> float:
        # Gini coefficient
        sorted_alloc = np.sort(allocations)
        n = len(sorted_alloc)
        index = np.arange(1, n + 1)
        total = np.sum(sorted_alloc)
        if total == 0:
            return 0.0
        return (2 * np.sum(index * sorted_alloc)) / (n * total) - (n + 1) / n
