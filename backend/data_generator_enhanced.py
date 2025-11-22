import numpy as np
import pandas as pd
from typing import Dict, Optional
from backend.data_generator import DataGenerator, TierType

class EnhancedDataGenerator(DataGenerator):
    """Enhanced data generator with scenario support"""
    
    @staticmethod
    def generate_users(n_users: int,
                      emergency_pct: float = 0.05,
                      premium_pct: float = 0.20,
                      seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate users with custom tier distribution
        
        Args:
            n_users: Number of users
            emergency_pct: Fraction of emergency users (0-1)
            premium_pct: Fraction of premium users (0-1)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate tier counts
        n_emergency = int(n_users * emergency_pct)
        n_premium = int(n_users * premium_pct)
        n_standard = n_users - n_emergency - n_premium
        
        # Generate tiers
        tiers = (
            [TierType.EMERGENCY.value] * n_emergency +
            [TierType.PREMIUM.value] * n_premium +
            [TierType.STANDARD.value] * n_standard
        )
        
        np.random.shuffle(tiers)
        
        users = []
        
        for i, tier in enumerate(tiers):
            if tier == TierType.EMERGENCY.value:
                # Emergency: High demand, high priority
                demand = np.random.uniform(50, 200)
                priority = np.random.uniform(9, 10)
                min_bw = demand * 0.9
                max_bw = demand * 1.5
                weight = 3.0
                type_code = 1
            elif tier == TierType.PREMIUM.value:
                # Premium: Medium-high demand
                demand = np.random.uniform(20, 100)
                priority = np.random.uniform(6, 8)
                min_bw = demand * 0.7
                max_bw = demand * 1.2
                weight = 2.0
                type_code = 2
            else:
                # Standard: Lower demand
                demand = np.random.uniform(5, 50)
                priority = np.random.uniform(1, 5)
                min_bw = demand * 0.3
                max_bw = demand * 1.0
                weight = 1.0
                type_code = 3
            
            users.append({
                'user_id': f"User_{i}",
                'tier': tier,
                'base_demand_mbps': demand,
                'priority': priority,
                'min_bandwidth_mbps': min_bw,
                'max_bandwidth_mbps': max_bw,
                'user_type_code': type_code,
                'allocation_weight': weight,
                'user_type_name': tier.title()
            })
        
        return pd.DataFrame(users)
    
    @staticmethod
    def generate_emergency_scenarios(users_df: pd.DataFrame,
                                     scenario: str = 'normal') -> Dict[str, float]:
        """
        Generate emergency scenario parameters
        
        Args:
            users_df: User DataFrame
            scenario: Scenario type ('normal', 'disaster', 'cyber_attack', 'mass_event', 'infrastructure_failure')
        
        Returns:
            Dictionary with multipliers for different tiers
        """
        scenarios = {
            'normal': {
                'emergency_demand_multiplier': 1.0,
                'premium_demand_multiplier': 1.0,
                'standard_demand_multiplier': 1.0,
                'capacity_multiplier': 1.0,
                'name': 'Normal Operations'
            },
            'disaster': {
                'emergency_demand_multiplier': 3.0,
                'premium_demand_multiplier': 0.8,
                'standard_demand_multiplier': 0.5,
                'capacity_multiplier': 0.8,
                'name': 'Natural Disaster'
            },
            'cyber_attack': {
                'emergency_demand_multiplier': 2.0,
                'premium_demand_multiplier': 1.5,
                'standard_demand_multiplier': 0.3,
                'capacity_multiplier': 0.6,
                'name': 'Cyber Attack'
            },
            'mass_event': {
                'emergency_demand_multiplier': 1.2,
                'premium_demand_multiplier': 2.5,
                'standard_demand_multiplier': 2.0,
                'capacity_multiplier': 0.9,
                'name': 'Mass Event'
            },
            'infrastructure_failure': {
                'emergency_demand_multiplier': 1.5,
                'premium_demand_multiplier': 1.0,
                'standard_demand_multiplier': 0.7,
                'capacity_multiplier': 0.5,
                'name': 'Infrastructure Failure'
            }
        }
        
        return scenarios.get(scenario, scenarios['normal'])
    
    @staticmethod
    def generate_time_varying_demands(users_df: pd.DataFrame,
                                      n_timesteps: int = 24,
                                      pattern: str = 'daily') -> pd.DataFrame:
        """
        Generate time-varying demands (hourly pattern)
        
        Args:
            users_df: Base user DataFrame
            n_timesteps: Number of time steps
            pattern: 'daily', 'weekend', 'business_hours'
        """
        patterns = {
            'daily': [0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.8, 0.8,
                     0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            'weekend': [0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4],
            'business_hours': [0.2, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2]
        }
        
        multipliers = patterns.get(pattern, patterns['daily'])[:n_timesteps]
        
        time_series = []
        
        for t, mult in enumerate(multipliers):
            df_t = users_df.copy()
            df_t['timestep'] = t
            df_t['hour'] = t
            df_t['base_demand_mbps'] *= mult
            df_t['min_bandwidth_mbps'] *= mult
            df_t['max_bandwidth_mbps'] *= mult
            time_series.append(df_t)
        
        return pd.concat(time_series, ignore_index=True)
