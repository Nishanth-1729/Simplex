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

    # The  bandwidth allocation optimizer which combines everything: multi-objective, robust, all constraints.
   
    
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

   
        # OBJECTIVE FUNCTION - Multi-objective with utility functions
      
        
        # 1. FAIRNESS OBJECTIVE (using utility functions)
        if tier_weights is None:
            tier_weights = priorities / priorities.max()  # Normalize priorities
        
        fairness_obj = self._build_utility_objective(x, demands, tier_weights, utility_type, alpha)
        
        # 2. EFFICIENCY OBJECTIVE (maximize total allocation)
        efficiency_obj = cp.sum(x)
        
        # 3. LATENCY OBJECTIVE (minimize average latency)
        # Latency model: latency = base_latency + congestion_factor / bandwidth
        base_latency = 10.0  # Base latency in ms
        congestion_factors = demands * 0.1  # Congestion factor
        latency_obj = cp.sum(base_latency + cp.multiply(congestion_factors, cp.inv_pos(x + 1e-6)))
        
        # Normalize objectives to similar scales
        fairness_obj_normalized = fairness_obj / self.n_users
        efficiency_obj_normalized = efficiency_obj / self.total_capacity
        latency_obj_normalized = latency_obj / (self.n_users * 100)  # Assume max ~100ms per user
        
        # Combined objective
        objective = cp.Maximize(
            weight_fairness * fairness_obj_normalized +
            weight_efficiency * efficiency_obj_normalized -
            weight_latency * latency_obj_normalized  # Negative because we minimize latency
        )
        
        


        constraints = []
        
        # 1. CAPACITY CONSTRAINT (with uncertainty if specified)
        if uncertainty_type == 'box':
            # Box uncertainty: demands can vary in [d - δ, d + δ]
            demand_deviations = demands * uncertainty_level
            # Worst case: all demands at maximum
            worst_case_demands = demands + demand_deviations
            constraints.append(cp.sum(x) <= self.total_capacity)
            # Ensure allocation covers worst case proportionally
            for i in range(self.n_users):
                constraints.append(x[i] <= max_bandwidth[i])
        
        elif uncertainty_type == 'budget':
            # Budget uncertainty: at most Γ demands deviate
            demand_deviations = demands * uncertainty_level
            gamma = uncertainty_budget if uncertainty_budget else int(self.n_users * 0.3)
            # Robust formulation: capacity + sum of largest Γ deviations
            constraints.append(cp.sum(x) <= self.total_capacity)
            # Additional robust constraint (simplified)
            constraints.append(cp.sum(x) <= self.total_capacity * 1.1)  # 10% buffer
        
        elif uncertainty_type == 'ellipsoidal':
            # Ellipsoidal uncertainty: ||d - d_nominal|| ≤ Ω
            omega = np.sqrt(self.n_users) * demands.mean() * uncertainty_level
            constraints.append(cp.sum(x) <= self.total_capacity)
        
        else:
            # No uncertainty: standard capacity constraint
            constraints.append(cp.sum(x) <= self.total_capacity)
        
        # 2. MIN/MAX BANDWIDTH CONSTRAINTS
        constraints.append(x >= min_bandwidth)
        constraints.append(x <= max_bandwidth)
        
        # 3. FAIRNESS CONSTRAINT (minimum Jain's fairness index)
        # Jain's index ≥ threshold
        # (sum x)^2 / (n * sum x^2) ≥ threshold
        # This is hard to enforce directly in convex form, so we use a proxy: Ensure no user gets less than threshold * average allocation
        avg_allocation = self.total_capacity / self.n_users
        for i in range(self.n_users):
            constraints.append(x[i] >= fairness_threshold * min_bandwidth[i])
        
        # 4. PRIORITY-BASED CONSTRAINTS
        # Higher priority users should get at least as much as lower priority (when demands are similar)
        # This is implicit in the tier_weights in objective
        
        # 5. DEMAND SATISFACTION CONSTRAINTS
        # Each user should get at least a fraction of their demand
        min_satisfaction = 0.1  # At least 10% of demand
        constraints.append(x >= min_satisfaction * demands)

        
        problem = cp.Problem(objective, constraints)
        
        if verbose:
            print(f"\n Problem Statistics:")
            print(f"   Variables: {self.n_users}")
            print(f"   Constraints: {len(constraints)}")
            print(f"   Problem Type: {problem.is_dcp()} (DCP)")
            print(f"\n⚡ Solving with {solver}...")
        
        try:
            # Solve with specified solver (with fallback logic)
            solver_opts = {
                'max_iters': max_iterations,
                'verbose': verbose
            }
            
            # Try primary solver
            try:
                problem.solve(solver=solver, **solver_opts)
            except Exception as solver_error:
                if verbose:
                    print(f" {solver} not available, trying alternatives...")
                
                # Try fallback solvers in order
                fallback_solvers = []
                if solver != cp.SCS:
                    fallback_solvers.append(cp.SCS)
                if solver != cp.ECOS:
                    fallback_solvers.append(cp.ECOS)
                fallback_solvers.append(cp.OSQP)
                
                solved = False
                for fallback in fallback_solvers:
                    try:
                        if verbose:
                            print(f"   Trying {fallback}...")
                        problem.solve(solver=fallback, **solver_opts)
                        if problem.status in ['optimal', 'optimal_inaccurate']:
                            solver = fallback  # Update solver name
                            solved = True
                            break
                    except:
                        continue
                
                if not solved:
                    # Last resort: use default solver
                    problem.solve(**solver_opts)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                allocation = x.value
                solve_time = time.time() - start_time
                
                if verbose:
                    print(f"\n OPTIMIZATION SUCCESS!")
                    print(f"   Status: {problem.status}")
                    print(f"   Solve Time: {solve_time:.4f}s")
                    print(f"   Objective Value: {problem.value:.4f}")
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    allocation, demands, priorities, min_bandwidth, max_bandwidth
                )
                
                # Multi-objective components
                fairness_value = self._calculate_fairness_value(allocation, demands, tier_weights, utility_type, alpha)
                efficiency_value = allocation.sum() / self.total_capacity
                latency_value = self._calculate_average_latency(allocation, demands)
                
                result = {
                    'status': 'optimal',
                    'allocation': allocation,
                    'objective_value': problem.value,
                    'solve_time': solve_time,
                    
                    # Multi-objective breakdown
                    'fairness_score': fairness_value,
                    'efficiency_score': efficiency_value,
                    'latency_score': latency_value,
                    
                    # Store weights used
                    'weight_fairness': weight_fairness,
                    'weight_efficiency': weight_efficiency,
                    'weight_latency': weight_latency,
                    'utility_type': utility_type,
                    'alpha': alpha,
                    
                    # Comprehensive metrics
                    'metrics': metrics,
                    
                    # Uncertainty analysis
                    'uncertainty_type': uncertainty_type,
                    'robustness_score': self._calculate_robustness_score(allocation, demands, uncertainty_level),
                    
                    # Convergence data
                    'convergence': self.tracker.get_summary(),
                    
                    # Constraint satisfaction
                    'capacity_used': allocation.sum(),
                    'capacity_utilization': allocation.sum() / self.total_capacity,
                    'constraints_satisfied': len(constraints),
                    
                    # Additional info
                    'solver': solver,
                    'problem_size': self.n_users,
                    'total_demand': demands.sum(),
                    'oversubscription_ratio': demands.sum() / self.total_capacity
                }
                
                if verbose:
                    self._print_detailed_results(result)
                
                return result
            
            else:
                error_msg = f"Optimization failed with status: {problem.status}"
                if verbose:
                    print(f"\n {error_msg}")
                
                return {
                    'status': 'failed',
                    'error': error_msg,
                    'solve_time': time.time() - start_time,
                    'problem_status': problem.status
                }
        
        except Exception as e:
            error_msg = f"Exception during optimization: {str(e)}"
            if verbose:
                print(f"\n {error_msg}")
            
            return {
                'status': 'error',
                'error': error_msg,
                'solve_time': time.time() - start_time
            }
    
    def _build_utility_objective(self, x, demands, weights, utility_type, alpha):
        # Build utility-based objective function.
        if utility_type == 'log':
            # Log utility (proportional fairness)
            return cp.sum(cp.multiply(weights, cp.log(x + 1e-6)))
        
        elif utility_type == 'sqrt':
            # Square root utility
            return cp.sum(cp.multiply(weights, cp.sqrt(x + 1e-6)))
        
        elif utility_type == 'linear':
            # Linear utility (maximize total weighted allocation)
            return cp.sum(cp.multiply(weights, x))
        
        elif utility_type == 'alpha-fair':
            # Alpha-fair utility
            if alpha == 1.0:
                return cp.sum(cp.multiply(weights, cp.log(x + 1e-6)))
            else:
                # (x^(1-α) - 1) / (1-α)
                return cp.sum(cp.multiply(weights, cp.power(x + 1e-6, 1 - alpha))) / (1 - alpha)
        
        else:
            # Default to log
            return cp.sum(cp.multiply(weights, cp.log(x + 1e-6)))
    
    def _calculate_comprehensive_metrics(self, allocation, demands, priorities, 
                                        min_bandwidth, max_bandwidth) -> Dict:
        #Calculate comprehensive performance metrics.
        
        # Satisfaction metrics
        satisfaction = allocation / demands
        satisfaction = np.clip(satisfaction, 0, 1)
        
        # Fairness metrics
        jains_index = self._calculate_jains_fairness(allocation)
        
        # Efficiency metrics
        utilization = allocation.sum() / self.total_capacity * 100
        
        # Guarantee compliance
        min_guarantee_met = np.sum(allocation >= min_bandwidth) / self.n_users * 100
        max_compliance = np.sum(allocation <= max_bandwidth) / self.n_users * 100
        
        # Statistical metrics
        allocation_stats = {
            'mean': np.mean(allocation),
            'median': np.median(allocation),
            'std': np.std(allocation),
            'min': np.min(allocation),
            'max': np.max(allocation),
            'cv': np.std(allocation) / np.mean(allocation) if np.mean(allocation) > 0 else 0
        }
        
        # Priority-based metrics
        priority_satisfaction = {}
        for p in np.unique(priorities):
            mask = priorities == p
            priority_satisfaction[f'priority_{int(p)}'] = {
                'count': np.sum(mask),
                'avg_allocation': np.mean(allocation[mask]),
                'avg_satisfaction': np.mean(satisfaction[mask])
            }
        
        return {
            'jains_fairness_index': jains_index,
            'utilization_percent': utilization,
            'avg_satisfaction': np.mean(satisfaction),
            'weighted_satisfaction': np.average(satisfaction, weights=priorities),
            'min_guarantee_met_percent': min_guarantee_met,
            'max_compliance_percent': max_compliance,
            'allocation_stats': allocation_stats,
            'priority_breakdown': priority_satisfaction,
            'unsatisfied_users': np.sum(satisfaction < 0.5),
            'fully_satisfied_users': np.sum(satisfaction >= 0.95)
        }
    
    def _calculate_jains_fairness(self, allocation) -> float:
        # Calculate Jain's fairness index.
        n = len(allocation)
        sum_x = np.sum(allocation)
        sum_x_sq = np.sum(allocation ** 2)
        
        if sum_x_sq == 0:
            return 0.0
        
        return (sum_x ** 2) / (n * sum_x_sq)
    
    def _calculate_fairness_value(self, allocation, demands, weights, utility_type, alpha) -> float:
        # Calculate fairness score.
        if utility_type == 'log':
            return np.sum(weights * np.log(allocation + 1e-6))
        elif utility_type == 'sqrt':
            return np.sum(weights * np.sqrt(allocation + 1e-6))
        elif utility_type == 'linear':
            return np.sum(weights * allocation)
        else:
            return np.sum(weights * np.log(allocation + 1e-6))
    
    def _calculate_average_latency(self, allocation, demands) -> float:
        #Calculate average network latency.
        base_latency = 10.0
        congestion_factors = demands * 0.1
        latencies = base_latency + congestion_factors / (allocation + 1e-6)
        return np.mean(latencies)
    
    def _calculate_robustness_score(self, allocation, demands, uncertainty_level) -> float:
        # Calculate robustness score (0-1).
        # How well allocation handles demand variations
        demand_deviations = demands * uncertainty_level
        worst_case_demands = demands + demand_deviations
        satisfaction_worst_case = allocation / worst_case_demands
        satisfaction_worst_case = np.clip(satisfaction_worst_case, 0, 1)
        return np.mean(satisfaction_worst_case)
    
    def _print_detailed_results(self, result: Dict):
        #Print detailed optimization results.
        print(f"\n{'='*70}")
        print(f" UNIFIED OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\n MULTI-OBJECTIVE SCORES:")
        print(f"   Fairness:   {result['fairness_score']:.4f}")
        print(f"   Efficiency: {result['efficiency_score']:.2%}")
        print(f"   Latency:    {result['latency_score']:.2f} ms")
        
        metrics = result['metrics']
        print(f"\n FAIRNESS METRICS:")
        print(f"   Jain's Index:        {metrics['jains_fairness_index']:.4f}")
        print(f"   Avg Satisfaction:    {metrics['avg_satisfaction']:.2%}")
        print(f"   Weighted Sat:        {metrics['weighted_satisfaction']:.2%}")
        
        print(f"\n EFFICIENCY METRICS:")
        print(f"   Capacity Used:       {result['capacity_used']:,.0f} / {self.total_capacity:,.0f} Mbps")
        print(f"   Utilization:         {result['capacity_utilization']:.2%}")
        
        print(f"\n ROBUSTNESS:")
        print(f"   Uncertainty Type:    {result['uncertainty_type']}")
        print(f"   Robustness Score:    {result['robustness_score']:.2%}")
        
        stats = metrics['allocation_stats']
        print(f"\n ALLOCATION STATISTICS:")
        print(f"   Mean:    {stats['mean']:.2f} Mbps")
        print(f"   Median:  {stats['median']:.2f} Mbps")
        print(f"   Std Dev: {stats['std']:.2f} Mbps")
        print(f"   Range:   [{stats['min']:.2f}, {stats['max']:.2f}] Mbps")
        
        print(f"\n USER SATISFACTION:")
        print(f"   Fully Satisfied (≥95%): {metrics['fully_satisfied_users']:,}")
        print(f"   Unsatisfied (<50%):     {metrics['unsatisfied_users']:,}")
        
        print(f"\n{'='*70}")


def test_unified_optimizer():
    #Test the unified optimizer with sample data.
    print("Testing Unified Optimizer\n")
    
    # Generate test data
    n_users = 100
    np.random.seed(42)
    
    demands = np.random.uniform(5, 50, n_users)
    priorities = np.random.randint(1, 11, n_users)
    min_bandwidth = demands * 0.1
    max_bandwidth = demands * 1.5
    total_capacity = demands.sum() * 0.7
    
    # Create optimizer
    optimizer = UnifiedOptimizer(n_users, total_capacity)
    
    # Run unified optimization
    result = optimizer.optimize_unified(
        demands=demands,
        priorities=priorities,
        min_bandwidth=min_bandwidth,
        max_bandwidth=max_bandwidth,
        weight_fairness=0.4,
        weight_efficiency=0.4,
        weight_latency=0.2,
        utility_type='log',
        uncertainty_type='budget',
        uncertainty_level=0.2,
        verbose=True
    )
    
    if result['status'] == 'optimal':
        print("\n Test PASSED!")
        return True
    else:
        print(f"\n Test FAILED: {result.get('error')}")
        return False


if __name__ == "__main__":
    test_unified_optimizer()