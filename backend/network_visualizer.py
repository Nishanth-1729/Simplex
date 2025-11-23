"""
Building a network visualization engine with real time and 3-D nettworks and paths highlights 
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import colorsys


class NetworkVisualizer:
    """
    Initialization 
    ULTIMATE Network Visualization Engine
    Creates beautiful(Implementing colours), interactive visualizations of network topologies
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.color_schemes = {
            'emergency': '#FF1744',  # Red
            'premium': '#2196F3',    # Blue
            'standard': '#4CAF50',   # Green
            'router1': '#FF9800',    # Orange
            'router2': '#9C27B0',    # Purple
            'source': '#FFC107'      # Amber
        }
    
    def create_network_topology_3d(self, optimizer, show_flows: bool = True,
                                   highlight_bottlenecks: bool = True) -> go.Figure:
        
        G = optimizer.graph
        
        # Get positions 
        pos_2d = {}
        for node_id, node_data in G.nodes(data=True):
            if 'pos' in node_data and node_data['pos']:
                pos_2d[node_id] = node_data['pos']
        
        if not pos_2d:
            pos_2d = nx.spring_layout(G, dim=2, k=2, iterations=50)
        
        # Convert to 3D 
        pos = {}
        for node_id in G.nodes():
            x, y = pos_2d.get(node_id, (0, 0))
            if node_id == "SOURCE":
                z = 3.0
            elif node_id.startswith("R1_"):
                z = 2.0
            elif node_id.startswith("R2_"):
                z = 1.0
            else:
                z = 0.0
            
            pos[node_id] = (x, y, z)
        
        fig = go.Figure()
        
        # Draw edges (links)
        edge_traces = self._create_edge_traces(
            G, pos, optimizer, show_flows, highlight_bottlenecks
        )
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Draw nodes
        node_traces = self._create_node_traces(G, pos, optimizer)
        for trace in node_traces:
            fig.add_trace(trace)
        
        # Add flow annotations if requested
        if show_flows:
            annotations = self._create_flow_annotations(G, pos, optimizer)
            fig.update_layout(scene=dict(annotations=annotations))
        
        # Layout
        fig.update_layout(
            title={
                'text': '<b> Network Topology - 3D Interactive View</b><br>' +
                        '<sub>Drag to rotate | Scroll to zoom | Click nodes for details</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial Black'}
            },
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                bgcolor='rgb(240,240,240)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=800,
            hovermode='closest'
        )
        
        return fig
    
    def _create_edge_traces(self, G, pos, optimizer, show_flows, highlight_bottlenecks):
        traces = []
        
        # Group edges by utilization for coloring
        edge_groups = {'low': [], 'medium': [], 'high': [], 'critical': []}
        
        for edge in G.edges():
            src, dst = edge
            
            link = optimizer.links.get(edge)
            if not link:
                continue
            utilization = link.current_load / link.capacity if link.capacity > 0 else 0
            
            # Categorize
            if utilization < 0.5:
                category = 'low'
            elif utilization < 0.7:
                category = 'medium'
            elif utilization < 0.9:
                category = 'high'
            else:
                category = 'critical'
            
            edge_groups[category].append((edge, utilization, link))
        
        # Create traces and add the colours for each category
        colors = {
            'low': 'rgba(76, 175, 80, 0.6)',      # Green
            'medium': 'rgba(255, 193, 7, 0.7)',   # Amber
            'high': 'rgba(255, 152, 0, 0.8)',     # Orange
            'critical': 'rgba(244, 67, 54, 0.9)'  # Red
        }
        
        widths = {'low': 2, 'medium': 3, 'high': 4, 'critical': 6}
        
        for category, edges in edge_groups.items():
            if not edges:
                continue
            
            x_coords = []
            y_coords = []
            z_coords = []
            hover_texts = []
            
            for (src, dst), util, link in edges:
                x0, y0, z0 = pos[src]
                x1, y1, z1 = pos[dst]
                
                x_coords.extend([x0, x1, None])
                y_coords.extend([y0, y1, None])
                z_coords.extend([z0, z1, None])
                
                hover_text = (f"<b>{src} → {dst}</b><br>" +
                            f"Capacity: {link.capacity:.0f} Mbps<br>" +
                            f"Load: {link.current_load:.0f} Mbps<br>" +
                            f"Utilization: {util:.1%}<br>" +
                            f"Latency: {link.latency:.1f} ms")
                hover_texts.extend([hover_text, hover_text, hover_text])
            
            trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color=colors[category],
                    width=widths[category]
                ),
                hovertext=hover_texts,
                hoverinfo='text',
                name=f'{category.title()} Load ({len(edges)} links)',
                showlegend=True
            )
            traces.append(trace)
        
        return traces
    
    def _create_node_traces(self, G, pos, optimizer):
        """Create node traces with proper styling"""
        traces = []
        
        # Group nodes by type
        node_groups = {
            'source': [],
            'router1': [],
            'router2': [],
            'emergency': [],
            'premium': [],
            'standard': []
        }
        
        for node_id in G.nodes():
            node = optimizer.nodes.get(node_id)
            if not node:
                continue
            
            if node.type.value == 'source':
                node_groups['source'].append(node_id)
            elif node.type.value == 'router':
                if node_id.startswith('R1_'):
                    node_groups['router1'].append(node_id)
                else:
                    node_groups['router2'].append(node_id)
            elif node.type.value == 'user':
                if node.qos_class:
                    qos_name = node.qos_class.name.lower()
                    node_groups[qos_name].append(node_id)
        
        # Create trace for each group
        node_config = {
            'source': {'color': self.color_schemes['source'], 'size': 25, 'symbol': 'diamond', 'name': ' Source'},
            'router1': {'color': self.color_schemes['router1'], 'size': 15, 'symbol': 'square', 'name': ' Core Routers'},
            'router2': {'color': self.color_schemes['router2'], 'size': 12, 'symbol': 'square', 'name': ' Edge Routers'},
            'emergency': {'color': self.color_schemes['emergency'], 'size': 8, 'symbol': 'circle', 'name': ' Emergency'},
            'premium': {'color': self.color_schemes['premium'], 'size': 7, 'symbol': 'circle', 'name': ' Premium'},
            'standard': {'color': self.color_schemes['standard'], 'size': 6, 'symbol': 'circle', 'name': ' Standard'}
        }
        
        for group_name, node_ids in node_groups.items():
            if not node_ids:
                continue
            
            config = node_config[group_name]
            
            x_coords = []
            y_coords = []
            z_coords = []
            hover_texts = []
            
            for node_id in node_ids:
                x, y, z = pos[node_id]
                node = optimizer.nodes[node_id]
                
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
                
                # Hover text
                hover_text = f"<b>{node_id}</b><br>"
                hover_text += f"Type: {node.type.value.title()}<br>"
                hover_text += f"Capacity: {node.capacity:.0f} Mbps<br>"
                hover_text += f"Delay: {node.processing_delay:.1f} ms"
                
                if node.qos_class:
                    hover_text += f"<br>QoS: {node.qos_class.name}"
                
                hover_texts.append(hover_text)
            
            trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers+text',
                marker=dict(
                    size=config['size'],
                    color=config['color'],
                    symbol=config['symbol'],
                    line=dict(color='white', width=2)
                ),
                text=[nid.split('_')[0] if '_' in nid else nid for nid in node_ids],
                textposition='top center',
                textfont=dict(size=8, color='black'),
                hovertext=hover_texts,
                hoverinfo='text',
                name=config['name'],
                showlegend=True
            )
            traces.append(trace)
        
        return traces
    
    def _create_flow_annotations(self, G, pos, optimizer):
        """Create annotations showing flow amounts"""
        annotations = []
        
        # Only annotate high-flow edges
        for edge, flow in optimizer.link_utilization.items():
            if flow < 100:  # Skip small flows
                continue
            
            src, dst = edge
            if src not in pos or dst not in pos:
                continue
            
            x0, y0, z0 = pos[src]
            x1, y1, z1 = pos[dst]
            
            # Midpoint
            xm, ym, zm = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
            
            # Note: 3D annotations are limited in Plotly
            # We'll create text traces instead
        
        return annotations
    
    def create_network_topology_2d(self, optimizer, show_flows: bool = True) -> go.Figure:
        G = optimizer.graph
        
        # Use spring/force-directed layout for natural graph appearance
        # This creates a more organic, non-hierarchical look
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42, scale=10)
        
        fig = go.Figure()
        
        # Draw edges with bandwidth allocation info
        for src, dst in G.edges():
            link = optimizer.links.get((src, dst))
            
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            
            # Determine edge color and width based on utilization
            if link and link.capacity > 0:
                utilization = link.current_load / link.capacity
                color = self._utilization_to_color(utilization)
                width = max(1, min(utilization * 8, 10))  # Scale width with utilization
                
                # Show allocated bandwidth after optimization
                hover_text = f"<b>{src} → {dst}</b><br>"
                hover_text += f"Allocated: <b>{link.current_load:.1f} Mbps</b><br>"
                hover_text += f"Capacity: {link.capacity:.1f} Mbps<br>"
                hover_text += f"Utilization: {utilization:.1%}<br>"
                hover_text += f"Latency: {link.latency:.1f} ms"
            else:
                color = 'rgba(100, 100, 100, 0.3)'
                width = 1
                hover_text = f"{src} → {dst}<br>No allocation"
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color=color, width=width),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False
            )
            fig.add_trace(edge_trace)
        
        # Categorize nodes by type for coloring
        node_groups = {
            'source': [],
            'router1': [],
            'router2': [],
            'emergency': [],
            'premium': [],
            'standard': []
        }
        
        for node_id in G.nodes():
            node = optimizer.nodes.get(node_id)
            if node_id == "SOURCE":
                node_groups['source'].append(node_id)
            elif node_id.startswith("R1_"):
                node_groups['router1'].append(node_id)
            elif node_id.startswith("R2_"):
                node_groups['router2'].append(node_id)
            elif node and hasattr(node, 'qos_class') and node.qos_class is not None:
                if node.qos_class.value == 1:
                    node_groups['emergency'].append(node_id)
                elif node.qos_class.value == 2:
                    node_groups['premium'].append(node_id)
                else:
                    node_groups['standard'].append(node_id)
            else:
                node_groups['standard'].append(node_id)
        
        # Draw nodes by type
        group_config = {
            'source': {'name': ' Source', 'color': self.color_schemes['source'], 'size': 35},
            'router1': {'name': ' Core Routers', 'color': self.color_schemes['router1'], 'size': 25},
            'router2': {'name': ' Edge Routers', 'color': self.color_schemes['router2'], 'size': 18},
            'emergency': {'name': ' Emergency Users', 'color': self.color_schemes['emergency'], 'size': 12},
            'premium': {'name': ' Premium Users', 'color': self.color_schemes['premium'], 'size': 10},
            'standard': {'name': ' Standard Users', 'color': self.color_schemes['standard'], 'size': 8}
        }
        
        for group_name, nodes in node_groups.items():
            if not nodes:
                continue
            
            config = group_config[group_name]
            node_x = []
            node_y = []
            node_text = []
            
            for node_id in nodes:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                
                node = optimizer.nodes.get(node_id)
                
                if node:
                    hover_text = f"<b>{node_id}</b><br>Type: {node.type.value}<br>Capacity: {node.capacity:.0f} Mbps"
                    if hasattr(node, 'qos_class') and node.qos_class is not None:
                        qos_names = {1: 'Emergency', 2: 'Premium', 3: 'Standard'}
                        hover_text += f"<br>QoS: {qos_names.get(node.qos_class.value, 'N/A')}"
                else:
                    hover_text = node_id
                
                node_text.append(hover_text)
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=config['size'],
                    color=config['color'],
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                hovertext=node_text,
                hoverinfo='text',
                name=config['name'],
                showlegend=True
            )
            fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title={
                'text': " 2D Network Graph - Force-Directed Layout",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1f77b4'}
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            height=700,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )
        
        return fig
    
    def create_congestion_heatmap(self, optimizer) -> go.Figure:
        """
        Create 2D heat map showing network congestion
        """
        G = optimizer.graph
        
        # Get 2D layout
        pos = nx.spring_layout(G, dim=2, k=2, iterations=50)
        
        fig = go.Figure()
        
        # Draw edges with color based on utilization
        for edge in G.edges():
            src, dst = edge
            link = optimizer.links.get(edge)
            
            if not link or link.capacity == 0:
                continue
            
            utilization = link.current_load / link.capacity
            
            # Color scale: green -> yellow -> orange -> red
            color = self._utilization_to_color(utilization)
            
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=max(2, utilization * 10)),
                hovertext=f"{src} → {dst}<br>Utilization: {utilization:.1%}",
                hoverinfo='text',
                showlegend=False
            ))
        
        # Draw nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node = optimizer.nodes.get(node_id)
            
            if not node:
                continue
            
            node_x.append(x)
            node_y.append(y)
            
            # Color by type
            if node.type.value == 'source':
                node_colors.append(self.color_schemes['source'])
                node_sizes.append(30)
            elif node.type.value == 'router':
                if node_id.startswith('R1_'):
                    node_colors.append(self.color_schemes['router1'])
                    node_sizes.append(20)
                else:
                    node_colors.append(self.color_schemes['router2'])
                    node_sizes.append(15)
            else:
                if node.qos_class:
                    qos_name = node.qos_class.name.lower()
                    node_colors.append(self.color_schemes[qos_name])
                else:
                    node_colors.append(self.color_schemes['standard'])
                node_sizes.append(10)
            
            node_text.append(node_id)
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color='white', width=2)
            ),
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title='<b> Network Congestion Heat Map</b>',
            showlegend=False,
            hovermode='closest',
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgb(250,250,250)'
        )
        
        return fig
    
    def _utilization_to_color(self, utilization: float) -> str:
        """Convert utilization percentage to color"""
        if utilization < 0.5:
            # Green to yellow
            r = int(utilization * 2 * 255)
            g = 255
            b = 0
        elif utilization < 0.8:
            # Yellow to orange
            progress = (utilization - 0.5) / 0.3
            r = 255
            g = int(255 * (1 - progress * 0.4))
            b = 0
        else:
            # Orange to red
            progress = (utilization - 0.8) / 0.2
            r = 255
            g = int(153 * (1 - progress))
            b = 0
        
        return f'rgb({r},{g},{b})'
    
    def create_path_visualization(self, optimizer, demand_id: str) -> go.Figure:
        """
        Visualize the path(s) used for a specific traffic demand
        """
        if demand_id not in optimizer.paths:
            return go.Figure().add_annotation(
                text=f"No paths found for {demand_id}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        paths = optimizer.paths[demand_id]
        G = optimizer.graph
        
        # Layout
        pos = nx.spring_layout(G, dim=2, k=2, iterations=50)
        
        fig = go.Figure()
        
        # Draw all edges (faded)
        for edge in G.edges():
            src, dst = edge
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='rgba(200,200,200,0.3)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Draw paths (highlighted)
        path_colors = ['rgb(255,0,0)', 'rgb(0,0,255)', 'rgb(0,255,0)']
        
        for i, path in enumerate(paths[:3]):  # Show top 3 paths
            path_x = []
            path_y = []
            
            for node in path:
                x, y = pos[node]
                path_x.append(x)
                path_y.append(y)
            
            # Calculate path metrics
            metrics = optimizer.compute_path_metrics(path)
            
            fig.add_trace(go.Scatter(
                x=path_x,
                y=path_y,
                mode='lines+markers',
                line=dict(color=path_colors[i], width=4),
                marker=dict(size=10),
                name=f"Path {i+1} ({metrics['latency']:.1f}ms)",
                hovertext=f"Latency: {metrics['latency']:.1f}ms<br>" +
                         f"Capacity: {metrics['capacity']:.0f} Mbps<br>" +
                         f"Reliability: {metrics['reliability']:.2%}",
                hoverinfo='text'
            ))
        
        # Draw all nodes
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=12, color='lightblue', line=dict(color='black', width=1)),
            text=list(G.nodes()),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'<b> Path Visualization: {demand_id}</b>',
            showlegend=True,
            hovermode='closest',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_metrics_dashboard(self, optimizer, result: Dict) -> go.Figure:
        """
        Create comprehensive metrics dashboard
        """
        metrics = result.get('metrics', {})
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                ' Link Utilization Distribution',
                ' Demand Satisfaction',
                ' Latency Analysis',
                ' Network Topology Stats'
            ),
            specs=[[{'type': 'histogram'}, {'type': 'indicator'}],
                   [{'type': 'box'}, {'type': 'table'}]]
        )
        
        # 1. Link utilization histogram
        utilizations = []
        for edge, link in optimizer.links.items():
            if link.capacity > 0:
                util = link.current_load / link.capacity * 100
                utilizations.append(util)
        
        fig.add_trace(
            go.Histogram(
                x=utilizations,
                nbinsx=20,
                marker_color='rgba(33, 150, 243, 0.7)',
                name='Utilization'
            ),
            row=1, col=1
        )
        
        # 2. Satisfaction gauge
        satisfaction_rate = metrics.get('satisfaction_rate', 0) * 100
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=satisfaction_rate,
                title={'text': "Demand Satisfaction (%)"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=1, col=2
        )
        
        # 3. Latency box plot
        latencies = []
        for demand in optimizer.traffic_demands:
            if demand.id in optimizer.paths and optimizer.paths[demand.id]:
                path = optimizer.paths[demand.id][0]
                metrics_path = optimizer.compute_path_metrics(path)
                latencies.append(metrics_path['latency'])
        
        if latencies:
            fig.add_trace(
                go.Box(
                    y=latencies,
                    name='Latency',
                    marker_color='rgb(255, 152, 0)',
                    boxmean='sd'
                ),
                row=2, col=1
            )
        
        # 4. Network stats table
        summary = optimizer.get_network_summary()
        
        table_data = [
            ['Total Nodes', summary['nodes']['total']],
            ['Total Links', summary['links']['total']],
            ['Avg Utilization', f"{metrics.get('avg_link_utilization', 0):.1%}"],
            ['Max Utilization', f"{metrics.get('max_link_utilization', 0):.1%}"],
            ['Congested Links', f"{metrics.get('congested_links', 0)}"],
            ['Total Demand', f"{summary['traffic']['total_demand_volume']:.0f} Mbps"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[[row[0] for row in table_data], [row[1] for row in table_data]],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="<b> Network Performance Dashboard</b>",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Utilization (%)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Links", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
        
        return fig
    
    def create_animated_flow_visualization(self, optimizer) -> go.Figure:
        """
        Create animated visualization showing data flow through network
        (Simplified - full animation requires more complex implementation)
        """
        G = optimizer.graph
        pos = nx.spring_layout(G, dim=2, k=2, iterations=50)
        
        # Create frames for animation
        frames = []
        
        # For now, create static visualization
        # Full animation would require time-series flow data
        
        fig = self.create_congestion_heatmap(optimizer)
        fig.update_layout(title='<b> Network Flow Animation</b>')
        
        return fig


def demo_visualization():
    """Demonstrating the network visualization capabilities"""
    from network_topology_optimizer import NetworkTopologyOptimizer
    
    print(" Network Visualization Demo\n")
    
    # Create and optimize network
    optimizer = NetworkTopologyOptimizer()
    user_ids = optimizer.build_hierarchical_network(
        n_routers_layer1=3,
        n_routers_layer2=6,
        n_users=30
    )
    optimizer.generate_traffic_demands(user_ids, total_traffic_gbps=10.0)
    result = optimizer.optimize_flows_multi_commodity(verbose=False)
    
    # Create visualizer
    viz = NetworkVisualizer()
    
    print("Creating visualizations...")
    
    # 3D topology
    fig_3d = viz.create_network_topology_3d(optimizer)
    print(" 3D topology created")
    
    # Congestion heatmap
    fig_heatmap = viz.create_congestion_heatmap(optimizer)
    print(" Congestion heatmap created")
    
    # Metrics dashboard
    fig_dashboard = viz.create_metrics_dashboard(optimizer, result)
    print(" Metrics dashboard created")
    
    print("\n All visualizations ready!")
    print(" Use fig.show() to display in browser")
    
    return viz, optimizer


if __name__ == "__main__":
    viz, optimizer = demo_visualization()
