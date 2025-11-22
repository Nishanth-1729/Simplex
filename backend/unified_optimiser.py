import cvxpy as cp
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
import time

class ConvergenceTracker:
    #Tracks optimization convergence in real-time.
    
    def __init__(self):
        self.iterations = []
        self.objective_values = []
        self.primal_residuals = []
        self.dual_residuals = []
        self.gaps = []
        self.constraint_violations = []
        self.timestamps = []
        self.start_time = None
    
    def reset(self):
       # Reset all tracking data.
        self.__init__()
    
    def add_iteration(self, iteration: int, obj_val: float, 
                     primal_res: float, dual_res: float,
                     gap: float, constraint_viol: float):
        #Add iteration data.
        if self.start_time is None:
            self.start_time = time.time()
        
        self.iterations.append(iteration)
        self.objective_values.append(obj_val)
        self.primal_residuals.append(primal_res)
        self.dual_residuals.append(dual_res)
        self.gaps.append(gap)
        self.constraint_violations.append(constraint_viol)
        self.timestamps.append(time.time() - self.start_time)
    
    def get_summary(self) -> Dict:
        #Gets convergence summary.
        if not self.iterations:
            return {}
        
        return {
            'total_iterations': len(self.iterations),
            'final_objective': self.objective_values[-1] if self.objective_values else None,
            'final_gap': self.gaps[-1] if self.gaps else None,
            'total_time': self.timestamps[-1] if self.timestamps else 0,
            'convergence_rate': self._calculate_convergence_rate(),
            'iterations': self.iterations,
            'objective_values': self.objective_values,
            'primal_residuals': self.primal_residuals,
            'dual_residuals': self.dual_residuals,
            'gaps': self.gaps,
            'constraint_violations': self.constraint_violations,
            'timestamps': self.timestamps
        }
    
    def _calculate_convergence_rate(self) -> float:
        #Calculate convergence rate aka improvement per iteration
        if len(self.objective_values) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(self.objective_values)):
            if self.objective_values[i-1] != 0:
                improvement = abs(self.objective_values[i] - self.objective_values[i-1]) / abs(self.objective_values[i-1])
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
class UnifiedOptimizer:

    # The ultimate bandwidth allocation optimizer which combines everything: multi-objective, robust, all constraints.
   
    
    def __init__(self, n_users: int, total_capacity: float):
       
        # Initialize the unified optimizer.
        
        # Args :  n_users: Number of users, total_capacity: Total network capacity (Mbps)
       
        self.n_users = n_users
        self.total_capacity = total_capacity
        self.tracker = ConvergenceTracker()
        self.logger = logging.getLogger(__name__)
    
    def optimize_unified(self,
                        demands: np.ndarray,
                        priorities: np.ndarray,
                        min_bandwidth: np.ndarray,
                        max_bandwidth: np.ndarray,

                        # Multi-objective weights
                        weight_fairness: float = 0.4,
                        weight_efficiency: float = 0.4,
                        weight_latency: float = 0.2,

                        # Utility function
                        utility_type: str = 'log',
                        alpha: float = 0.5,

                        # Robust optimization
                        uncertainty_type: Optional[str] = 'budget',
                        uncertainty_level: float = 0.2,
                        uncertainty_budget: Optional[int] = None,

                        # Additional constraints
                        tier_weights: Optional[np.ndarray] = None,
                        fairness_threshold: float = 0.7,

                        # Solver options
                        verbose: bool = True,
                        max_iterations: int = 10000,
                        solver: str = 'ECOS') -> Dict:
     
        # The final opttimisation , everything combined
        
        # Args:
        #     demands: User bandwidth demands (Mbps)
        #     priorities: User priority levels (1-10)
        #     min_bandwidth: Minimum bandwidth guarantees
        #     max_bandwidth: Maximum bandwidth limits
        #     weight_fairness: Weight for fairness objective (0-1)
        #     weight_efficiency: Weight for efficiency objective (0-1)
        #     weight_latency: Weight for latency objective (0-1)
        #     utility_type: Utility function ('log', 'sqrt', 'linear', 'alpha-fair')
        #     alpha: Alpha parameter for alpha-fair utility
        #     uncertainty_type: Type of uncertainty ('box', 'budget', 'ellipsoidal', None)
        #     uncertainty_level: Fraction of demand that can deviate (0-1)
        #     uncertainty_budget: Number of users that can deviate (for budget uncertainty)
        #     tier_weights: Optional tier-based allocation weights
        #     fairness_threshold: Minimum fairness index required
        #     verbose: Print optimization progress
        #     max_iterations: Maximum solver iterations
        #     solver: CVXPY solver to use
        
        # Returns: Comprehensive optimization results with convergence data
     
        start_time = time.time()
        self.tracker.reset()
        
        if verbose:
            print(" UNIFIED OPTIMIZER - FULL POWER MODE ACTIVATED!")
            print(f"   Users: {self.n_users:,}")
            print(f"   Capacity: {self.total_capacity:,.0f} Mbps")
            print(f"   Total Demand: {demands.sum():,.0f} Mbps")
            print(f"   Oversubscription: {demands.sum()/self.total_capacity:.2f}x")
            print(f"   Multi-Objective: Fairness={weight_fairness}, Efficiency={weight_efficiency}, Latency={weight_latency}")
            print(f"   Utility: {utility_type}")
            print(f"   Uncertainty: {uncertainty_type} (level={uncertainty_level})")
        
        # Normalize weights
        total_weight = weight_fairness + weight_efficiency + weight_latency
        if total_weight > 0:
            weight_fairness /= total_weight
            weight_efficiency /= total_weight
            weight_latency /= total_weight
        
        # Decision variable: bandwidth allocation
        x = cp.Variable(self.n_users, pos=True)
    
