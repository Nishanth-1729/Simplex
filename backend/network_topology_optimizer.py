from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

class NodeType(Enum):
    SOURCE = "source"
    ROUTER_L1 = "router_l1"
    ROUTER_L2 = "router_l2"
    USER = "user"
class QoSClass(Enum):
    EMERGENCY = 1
    PREMIUM = 2
    STANDARD = 3
@dataclass
class NetworkNode:
    id: str
    node_type: NodeType
    capacity: float
    qos_class: Optional[QoSClass] = None
    def _hash_(self):
        return hash(self.id)
@dataclass
class NetworkLink:
    source: str
    destination: str
    capacity: float
    latency: float
    current_load: float = 0.0
    def utilization(self) -> float:
        return (self.current_load / self.capacity * 100) if self.capacity > 0 else 0.0
    def available_capacity(self) -> float:
        return max(0.0, self.capacity - self.current_load)
@dataclass
class TrafficDemand:
    id: str
    source: str
    destination: str
    demand: float
    qos_class: QoSClass
    max_latency: float
    allocated: float = 0.0
    def satisfaction(self) -> float:
        return (self.allocated / self.demand * 100) if self.demand > 0 else 0.0
class NetworkTopologyOptimizer:
    def _init_(self, enable_redundancy: bool = True, enable_load_balancing: bool = True):
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[Tuple[str, str], NetworkLink] = {}
        self.traffic_demands: List[TrafficDemand] = []
        self.enable_redundancy = enable_redundancy
        self.enable_load_balancing = enable_load_balancing 
    def add_node(self, node: NetworkNode):
        self.nodes[node.id] = node
    def add_link(self, link: NetworkLink):
        self.links[(link.source, link.destination)] = link 
    def add_traffic_demand(self, demand: TrafficDemand):
        self.traffic_demands.append(demand)
    def build_hierarchical_network(
        self,
        n_routers_layer1: int,
        n_routers_layer2: int,
        n_users: int,
        source_capacity: float,
        router1_capacity: float,
        router2_capacity: float
    ) -> List[str]:
        "
        This is the building the hierarchical 4 layer network 
        Structure: Source → Core Routers → Edge Routers → Users
        "
        print(f"\n{'='*60}")
        print(f" BUILDING HIERARCHICAL NETWORK")
        print(f"{'='*60}")
        
        # Layer 0: Source
        source = NetworkNode(
            id="Source",
            node_type=NodeType.SOURCE,
            capacity=source_capacity
        )
        self.add_node(source)
        print(f" Created Source (Capacity: {source_capacity:.0f} Mbps)")
        
        # Layer 1 contains the Core Routers
        for i in range(n_routers_layer1):
            router = NetworkNode(
                id=f"R1_{i}",
                node_type=NodeType.ROUTER_L1,
                capacity=router1_capacity
            )
            self.add_node(router)
            
            # Connecting the  source to core router
            link = NetworkLink(
                source="Source",
                destination=f"R1_{i}",
                capacity=source_capacity / n_routers_layer1,
                latency=np.random.uniform(1.0, 5.0)
            )
            self.add_link(link)
        
        print(f" Created {n_routers_layer1} Core Routers (L1)")
        
        # Layer 2 containing the Edge Routers
        for i in range(n_routers_layer2):
            router = NetworkNode(
                id=f"R2_{i}",
                node_type=NodeType.ROUTER_L2,
                capacity=router2_capacity
            )
            self.add_node(router)
            
            # Connecting to all core routers
            for j in range(n_routers_layer1):
                link = NetworkLink(
                    source=f"R1_{j}",
                    destination=f"R2_{i}",
                    capacity=router1_capacity / n_routers_layer2,
                    latency=np.random.uniform(2.0, 8.0)
                )
                self.add_link(link)
        
        print(f" Created {n_routers_layer2} Edge Routers (L2)")
        
        # Layer 3 contains the Users
        user_ids = []
        users_per_router = n_users // n_routers_layer2
        
        for i in range(n_users):
            router_idx = i % n_routers_layer2
            user_id = f"User_{i}"
            
            user = NetworkNode(
                id=user_id,
                node_type=NodeType.USER,
                capacity=100.0
            )
            self.add_node(user)
            user_ids.append(user_id)
            
            # Connecting  to the  edge router
            link = NetworkLink(
                source=f"R2_{router_idx}",
                destination=user_id,
                capacity=router2_capacity / users_per_router,
                latency=np.random.uniform(1.0, 3.0)
            )
            self.add_link(link)
        
        print(f" Created {n_users} Users")
        print(f" Total: {len(self.nodes)} nodes, {len(self.links)} links")
        print(f"{'='*60}\n")
        
        return user_ids
    
    def get_network_summary(self) -> Dict:
        """Get network statistics."""
        node_counts = {}
        for node in self.nodes.values():
            node_counts[node.node_type.value] = node_counts.get(node.node_type.value, 0) + 1
        
        total_node_capacity = sum(n.capacity for n in self.nodes.values())
        total_link_capacity = sum(l.capacity for l in self.links.values())
        
        return {
            'nodes': {
                'total': len(self.nodes),
                'by_type': node_counts
            },
            'links': {
                'total': len(self.links)
            },
            'capacity': {
                'total_node_capacity': total_node_capacity,
                'total_link_capacity': total_link_capacity
            },
            'traffic': {
                'total_demand_volume': sum(d.demand for d in self.traffic_demands)
            }
        }
