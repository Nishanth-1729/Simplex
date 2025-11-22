import numpy as np
import pandas as pd
from enum import Enum

class TierType(Enum):
    # User tiers
    EMERGENCY = "emergency"
    PREMIUM = "premium"
    STANDARD = "standard"

class DataGenerator:
    # Creates realistic user data
    
    @staticmethod
    def generate_users(n_users: int) -> pd.DataFrame:
        # Create users with demands and priorities
        
        # Tier split
        n_emergency = int(n_users * 0.05)
        n_premium = int(n_users * 0.20)
        n_standard = n_users - n_emergency - n_premium
        
        tiers = (
            [TierType.EMERGENCY.value] * n_emergency +
            [TierType.PREMIUM.value] * n_premium +
            [TierType.STANDARD.value] * n_standard
        )
        
        np.random.shuffle(tiers)
        
        # Demand and limits
        demands = []
        priorities = []
        min_bw = []
        max_bw = []
        
        for tier in tiers:
            if tier == TierType.EMERGENCY.value:
                demand = np.random.uniform(50, 200)
                priority = np.random.uniform(9, 10)
                min_bw.append(demand * 0.9)
                max_bw.append(demand * 1.5)
            elif tier == TierType.PREMIUM.value:
                demand = np.random.uniform(20, 100)
                priority = np.random.uniform(6, 8)
                min_bw.append(demand * 0.7)
                max_bw.append(demand * 1.2)
            else:  # STANDARD
                demand = np.random.uniform(5, 50)
                priority = np.random.uniform(1, 5)
                min_bw.append(demand * 0.3)
                max_bw.append(demand * 1.0)
            
            demands.append(demand)
            priorities.append(priority)
        
        # Final dataset
        df = pd.DataFrame({
            'user_id': [f"User_{i}" for i in range(n_users)],
            'tier': tiers,
            'base_demand_mbps': demands,
            'priority': priorities,
            'min_bandwidth_mbps': min_bw,
            'max_bandwidth_mbps': max_bw,
            'user_type_code': [1 if t == TierType.EMERGENCY.value else 
                              2 if t == TierType.PREMIUM.value else 3 
                              for t in tiers],
            'allocation_weight': [3.0 if t == TierType.EMERGENCY.value else 
                                 2.0 if t == TierType.PREMIUM.value else 1.0 
                                 for t in tiers],
            'user_type_name': [t.title() for t in tiers]
        })
        
        return df
