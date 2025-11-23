"""
Network Visualization Module-3D/2D network topology visualization
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Optional

class NetworkVisualizer:
    """Createing the beautiful network topology visualizations"""
    
    @staticmethod
    def create_network_topology_3d(optimizer, 
                                   show_flows: bool = True,
                                   highlight_bottlenecks: bool = True) -> go.Figure:
        """
        Create 3D network topology visualization
        Args:
            optimizer: NetworkTopologyOptimizer instance
            show_flows: Show flow arrows
            highlight_bottlenecks: Highlight congested links
        """
        # Extracting the  node positions
        node_positions = {}
        node_colors = []
        node_sizes = []
        node_texts = []
        for node_id, node in optimizer.nodes.items():
            if hasattr(node, 'coordinates') and node.coordinates:
                x, y = node.coordinates
            else:
                # Use spring layout if no coordinates
                pos = nx.spring_layout(optimizer.graph, dim=3, seed=42)
                if node_id in pos:
                    x, y, z = pos[node_id]
                else:
                    x, y, z = 0, 0, 0
            
            # Assign z based on node type (layer)
            if node.type.value == 'source':
                z = 3
                color = '#FF1744'
                size = 30
            elif node.type.value == 'router':
                # Distinguish router layers by ID
                if 'R1_' in node_id:
                    z = 2
                    color = '#2196F3'
                    size = 20
                else:
                    z = 1
                    color = '#03A9F4'
                    size = 15
            else:  # user
                z = 0
                # Color by QoS if available
                if hasattr(node, 'qos_class') and node.qos_class:
                    if node.qos_class.value == 1:
                        color = '#FF6B6B'
                    elif node.qos_class.value == 2:
                        color = '#4ECDC4'
                    else:
                        color = '#95E1D3'
                else:
                    color = '#95E1D3'
                size = 10
            
            node_positions[node_id] = (x, y, z)
            node_colors.append(color)
            node_sizes.append(size)
            node_texts.append(f"{node_id}<br>Capacity: {node.capacity:.0f} Mbps")
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_z = []
        edge_colors = []
        
        for (src, dst), link in optimizer.links.items():
            if src in node_positions and dst in node_positions:
                x0, y0, z0 = node_positions[src]
                x1, y1, z1 = node_positions[dst]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                
                # Color by utilization
                if link.capacity > 0:
                    util = link.current_load / link.capacity
                    if util > 0.8:
                        color = 'red'
                    elif util > 0.5:
                        color = 'orange'
                    else:
                        color = 'green'
                else:
                    color = 'gray'
                
                edge_colors.append(color)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='lightgray', width=2),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        node_x = [node_positions[nid][0] for nid in optimizer.nodes.keys()]
        node_y = [node_positions[nid][1] for nid in optimizer.nodes.keys()]
        node_z = [node_positions[nid][2] for nid in optimizer.nodes.keys()]
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=2)
            ),
            text=list(optimizer.nodes.keys()),
            textposition='top center',
            hovertext=node_texts,
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title='3D Network Topology',
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            showlegend=False,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_network_topology_2d(optimizer, show_flows: bool = True) -> go.Figure:
        """Create 2D force-directed network layout"""
        
        pos = nx.spring_layout(optimizer.graph, k=0.5, iterations=50, seed=42)
        
        # Extract edges
        edge_trace = []
        
        for (src, dst), link in optimizer.links.items():
            if src in pos and dst in pos:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                
                # Color by utilization
                if link.capacity > 0:
                    util = link.current_load / link.capacity
                    width = max(1, util * 5)
                    
                    if util > 0.8:
                        color = 'red'
                    elif util > 0.5:
                        color = 'orange'
                    else:
                        color = 'green'
                else:
                    color = 'gray'
                    width = 1
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Extract nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node_id, node in optimizer.nodes.items():
            if node_id in pos:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                if node.type.value == 'source':
                    color = '#FF1744'
                    size = 30
                elif node.type.value == 'router':
                    color = '#2196F3'
                    size = 20
                else:
                    color = '#4CAF50'
                    size = 10
                
                node_colors.append(color)
                node_sizes.append(size)
                node_texts.append(f"{node_id}<br>{node.capacity:.0f} Mbps")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
            text=list(optimizer.nodes.keys()),
            textposition='top center',
            hovertext=node_texts,
            hoverinfo='text',
            showlegend=False
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='2D Network Graph',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_congestion_heatmap(optimizer) -> go.Figure:
        """Create link congestion heatmap"""
        
        links = []
        utilizations = []
        
        for (src, dst), link in optimizer.links.items():
            if link.capacity > 0:
                util = link.current_load / link.capacity
                links.append(f"{src}→{dst}")
                utilizations.append(util * 100)
        
        fig = go.Figure(data=go.Bar(
            x=links[:50],  # Show first 50 links
            y=utilizations[:50],
            marker=dict(
                color=utilizations[:50],
                colorscale='RdYlGn_r',
                cmin=0,
                cmax=100,
                colorbar=dict(title="Utilization (%)")
            )
        ))
        
        fig.update_layout(
            title='Link Utilization Heatmap',
            xaxis_title='Links',
            yaxis_title='Utilization (%)',
            height=400
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="80% threshold")
        
        return fig
    
    @staticmethod
    def create_metrics_dashboard(optimizer, result: Dict) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        
        metrics = result.get('metrics', {})
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'bar'}, {'type': 'pie'}]],
            subplot_titles=['Avg Utilization', 'Satisfaction Rate',
                           'Link Congestion', 'QoS Distribution']
        )
        
        # Gauge 1: Utilization
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('avg_link_utilization', 0) * 100,
            title={'text': "Avg Utilization (%)"},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "darkblue"},
                  'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 80}}
        ), row=1, col=1)
        
        # Gauge 2: Satisfaction
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('satisfaction_rate', 0) * 100,
            title={'text': "Satisfaction (%)"},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "green"}}
        ), row=1, col=2)
        
        fig.update_layout(height=600)
        
        return fig
