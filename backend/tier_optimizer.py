import cvxpy as cp
import numpy as np
from typing import Dict, Optional
import time

class TierBasedOptimizer:
    # Optimize bandwidth using user tiers
    
    def __init__(self, total_capacity: float):
        self.total_capacity = total_capacity
    
    def optimize_with_tiers(
        self,
        demands: np.ndarray,
        priorities: np.ndarray,
        min_bandwidth: np.ndarray,
        max_bandwidth: np.ndarray,
        tiers: np.ndarray,
        allocation_weights: np.ndarray,
        utility_type: str = "log"
    ) -> Dict:
        # Main optimizer with tier rules
        
        start_time = time.time()
        
        n_users = len(demands)
        
        # Bandwidth for each user
        x = cp.Variable(n_users, nonneg=True)
        
        # Utility based on chosen function
        if utility_type == "log":
            utility = cp.sum(cp.multiply(allocation_weights, cp.log(x + 1e-6)))
        elif utility_type == "sqrt":
            utility = cp.sum(cp.multiply(allocation_weights, cp.sqrt(x)))
        else:  # linear
            utility = cp.sum(cp.multiply(allocation_weights, x))
        
        objective = cp.Maximize(utility)
        
        # Basic limits
        constraints = [
            cp.sum(x) <= self.total_capacity,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x <= demands
        ]
        
        # Extra guarantees per tier
        for tier_val in [1, 2, 3]:
            tier_mask = (tiers == tier_val)
            if np.any(tier_mask):
                tier_demand = demands[tier_mask]
                tier_alloc = x[tier_mask]
                
                # Minimum share for each tier
                if tier_val == 1:  # Emergency: 90%
                    constraints.append(cp.sum(tier_alloc) >= 0.9 * cp.sum(tier_demand))
                elif tier_val == 2:  # Premium: 70%
                    constraints.append(cp.sum(tier_alloc) >= 0.7 * cp.sum(tier_demand))
                else:  # Standard: 30%
                    constraints.append(cp.sum(tier_alloc) >= 0.3 * cp.sum(tier_demand))
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                problem.solve(solver=cp.SCS, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                allocation = x.value
                
                # Per-tier summary
                tier_stats = {}
                for tier_val, tier_name in [(1, 'emergency'), (2, 'premium'), (3, 'free')]:
                    mask = (tiers == tier_val)
                    if np.any(mask):
                        tier_demand = demands[mask]
                        tier_alloc = allocation[mask]
                        tier_min = min_bandwidth[mask]
                        
                        tier_stats[tier_name] = {
                            'count': int(np.sum(mask)),
                            'total_demand': float(np.sum(tier_demand)),
                            'total_allocated': float(np.sum(tier_alloc)),
                            'avg_allocation': float(np.mean(tier_alloc)),
                            'avg_satisfaction': float(np.mean(tier_alloc / tier_demand)),
                            'guarantee_met_pct': float(np.sum(tier_alloc >= tier_min) / np.sum(mask) * 100)
                        }
                    else:
                        tier_stats[tier_name] = None
                
                # Overall fairness and usage
                jains_index = self._jains_index(allocation)
                
                return {
                    'status': 'optimal',
                    'allocation': allocation,
                    'objective_value': float(problem.value),
                    'total_allocated': float(np.sum(allocation)),
                    'efficiency': float(np.sum(allocation) / self.total_capacity),
                    'jains_fairness_index': float(jains_index),
                    'avg_satisfaction': float(np.mean(allocation / demands)),
                    'tier_statistics': tier_stats,
                    'solve_time': time.time() - start_time
                }
            else:
                return {
                    'status': problem.status,
                    'error': f'Optimization failed: {problem.status}'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def optimize_emergency_scenario(
        self,
        demands: np.ndarray,
        priorities: np.ndarray,
        min_bandwidth: np.ndarray,
        max_bandwidth: np.ndarray,
        tiers: np.ndarray,
        allocation_weights: np.ndarray,
        scenario_multipliers: Dict
    ) -> Dict:
        # Run optimization under emergency changes
        
        # Change demands by tier
        adjusted_demands = demands.copy()
        for i, tier in enumerate(tiers):
            if tier == 1:  # Emergency
                adjusted_demands[i] *= scenario_multipliers.get('emergency_demand_multiplier', 1.0)
            elif tier == 2:  # Premium
                adjusted_demands[i] *= scenario_multipliers.get('premium_demand_multiplier', 1.0)
            else:  # Standard
                adjusted_demands[i] *= scenario_multipliers.get('standard_demand_multiplier', 1.0)
        
        # Change network capacity
        adjusted_capacity = self.total_capacity * scenario_multipliers.get('capacity_multiplier', 1.0)
        
        # Store and update capacity
        original_capacity = self.total_capacity
        self.total_capacity = adjusted_capacity
        
        # Run main optimizer with adjusted values
        result = self.optimize_with_tiers(
            adjusted_demands, priorities, min_bandwidth, max_bandwidth,
            tiers, allocation_weights
        )
        
        # Restore original capacity
        self.total_capacity = original_capacity
        
        # Add extra info about this scenario
        result['adjusted_capacity'] = adjusted_capacity
        result['capacity_reduction'] = 1 - scenario_multipliers.get('capacity_multiplier', 1.0)
        result['emergency_capacity_used'] = float(np.sum(result.get('allocation', np.array([]))[tiers == 1]))
        
        return result
    
    def _jains_index(self, allocation: np.ndarray) -> float:
        # Calculate Jain's fairness index
        n = len(allocation)
        sum_x = np.sum(allocation)
        sum_x2 = np.sum(allocation ** 2)
        return float((sum_x ** 2) / (n * sum_x2)) if sum_x2 > 0 else 0.0
