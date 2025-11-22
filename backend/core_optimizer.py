"""
Core Optimizer Module
Person 3: Fundamental optimization
"""

import cvxpy as cp
import numpy as np
import time

class CoreOptimizer:
    # Main bandwidth optimizer
    
    def __init__(self, n_users: int, total_capacity: float):
        self.n_users = n_users
        self.total_capacity = total_capacity
    
    def optimize(
        self,
        demands: np.ndarray,
        priorities: np.ndarray,
        min_bandwidth: np.ndarray,
        max_bandwidth: np.ndarray,
        utility_type: str = "log",
        alpha: float = 1.0
    ) -> dict:
        # Runs optimization
        start_time = time.time()
        
        # Allocation variable
        x = cp.Variable(self.n_users, nonneg=True)
        
        # Choose utility function
        if utility_type == "log":
            utility = cp.sum(cp.log(x + 1e-6))
        elif utility_type == "sqrt":
            utility = cp.sum(cp.sqrt(x))
        elif utility_type == "linear":
            utility = cp.sum(x)
        else:  # alpha-fair
            if abs(alpha - 1.0) < 1e-6:
                utility = cp.sum(cp.log(x + 1e-6))
            else:
                utility = cp.sum(cp.power(x + 1e-6, 1 - alpha)) / (1 - alpha)
        
        objective = cp.Maximize(utility)
        
        # Constraints
        constraints = [
            cp.sum(x) <= self.total_capacity,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x <= demands
        ]
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status == cp.OPTIMAL:
            allocation = x.value
            
            # Jain's index
            sum_x = np.sum(allocation)
            sum_x2 = np.sum(allocation ** 2)
            jains_index = (sum_x ** 2) / (self.n_users * sum_x2) if sum_x2 > 0 else 0
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': time.time() - start_time,
                'utilization': np.sum(allocation) / self.total_capacity * 100,
                'metrics': {
                    'jains_fairness_index': jains_index
                }
            }
        else:
            return {'status': problem.status, 'error': 'Failed'}

class FairnessMetrics:
    # Fairness functions
    
    @staticmethod
    def jains_index(allocations: np.ndarray) -> float:
        # Jain's index
        n = len(allocations)
        sum_x = np.sum(allocations)
        sum_x2 = np.sum(allocations ** 2)
        return (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 0.0
    
    @staticmethod
    def gini_coefficient(allocations: np.ndarray) -> float:
        # Gini index
        sorted_alloc = np.sort(allocations)
        n = len(sorted_alloc)
        index = np.arange(1, n + 1)
        total = np.sum(sorted_alloc)
        if total == 0:
            return 0.0
        return (2 * np.sum(index * sorted_alloc)) / (n * total) - (n + 1) / n
