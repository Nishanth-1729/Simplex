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
