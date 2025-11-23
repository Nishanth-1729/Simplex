
#Comprehensive visualization tools for bandwidth allocation analysis.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class BandwidthVisualizer:
     # Advanced visualization tools for bandwidth allocation results.
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        # Initialize visualizer with matplotlib style.
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_allocation_comparison(self, allocations: Dict[str, np.ndarray], demands: np.ndarray, user_labels: List[str] = None, n_users_to_show: int = 20) -> go.Figure:
      
        # Compare different allocation strategies.
        # Args: allocations: Dictionary of {strategy_name: allocation_array},
        #      demands: User demands
        #      n_users_to_show: Number of users to display
        # Returns: Plotly figure
      
        # Select subset of users
        user_indices = np.linspace(0, len(demands) - 1, n_users_to_show, dtype=int)
        
        if user_labels is None:
            user_labels = [f'User {i+1}' for i in user_indices]
        
        fig = go.Figure()
        
        # Plot demands
        fig.add_trace(go.Bar(
            x=user_labels,
            y=demands[user_indices],
            name='Demand',
            marker_color='lightgray',
            opacity=0.6
        ))
        
        # Plot each allocation strategy
        for strategy_name, allocation in allocations.items():
            fig.add_trace(go.Bar(
                x=user_labels,
                y=allocation[user_indices],
                name=strategy_name
            ))
        
        fig.update_layout(
            title='Bandwidth Allocation Comparison',
            xaxis_title='Users',
            yaxis_title='Bandwidth (Mbps)',
            barmode='group',
            height=500,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    
    def plot_fairness_metrics(self, metrics: Dict[str, Dict]) -> go.Figure:
        
        # Visualize fairness metrics for different strategies.
        # Args: metrics: Dictionary of {strategy_name: metrics_dict}
        # Returns: Plotly figure
        
        strategies = list(metrics.keys())
        
        # Extract fairness metrics
        jains_index = [metrics[s].get('jains_fairness_index', 0) for s in strategies]
        gini_coef = [metrics[s].get('gini_coefficient', 0) for s in strategies]
        cv = [metrics[s].get('coefficient_of_variation', 0) for s in strategies]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Jain's Fairness Index", "Gini Coefficient", "Coefficient of Variation")
        )
        
        # Jain's index (higher is better)
        fig.add_trace(
            go.Bar(x=strategies, y=jains_index, name="Jain's Index", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Gini coefficient (lower is better)
        fig.add_trace(
            go.Bar(x=strategies, y=gini_coef, name="Gini Coefficient", marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Coefficient of variation (lower is better)
        fig.add_trace(
            go.Bar(x=strategies, y=cv, name="CV", marker_color='lightgreen'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="Fairness Metrics Comparison",
            showlegend=False,
            height=400
        )
        return fig
    
    def plot_pareto_frontier(self, fairness_values: List[float], efficiency_values: List[float], latency_values: List[float] = None) -> go.Figure:
       
        # Plot 2D or 3D Pareto frontier. Returns: Plotly figure
       
        if latency_values is None:
            # 2D plot: Fairness vs Efficiency
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fairness_values,
                y=efficiency_values,
                mode='markers+lines',
                marker=dict(size=10, color='blue'),
                line=dict(color='lightblue', width=2),
                name='Pareto Frontier'
            ))
            
            fig.update_layout(
                title='Pareto Frontier: Fairness vs Efficiency',
                xaxis_title='Fairness (Jain\'s Index)',
                yaxis_title='Efficiency (Utilization)',
                height=500
            )
        else:
            # 3D plot: Fairness vs Efficiency vs Latency
            fig = go.Figure(data=[go.Scatter3d(
                x=fairness_values,
                y=efficiency_values,
                z=latency_values,
                mode='markers',
                marker=dict(
                    size=8,
                    color=efficiency_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Efficiency")
                )
            )])
            
            fig.update_layout(
                title='3D Pareto Frontier',
                scene=dict(
                    xaxis_title='Fairness',
                    yaxis_title='Efficiency',
                    zaxis_title='Latency (ms)'
                ),
                height=600
            )
        
        return fig
    
    def plot_temporal_heatmap(self, allocation: np.ndarray, user_indices: List[int] = None, time_labels: List[str] = None) -> go.Figure:
        
        # Creates heatmap of temporal allocation.
        # Args: allocation: Array of shape (n_users, time_slots)
        #       user_indices: Subset of users to display
        #       time_labels: Labels for time slots
        # Returns: Plotly figure
        
        if user_indices is None:
            # Show first 50 users
            user_indices = list(range(min(50, allocation.shape[0])))
        
        if time_labels is None:
            time_labels = [f'{h:02d}:00' for h in range(allocation.shape[1])]
        
        user_labels = [f'User {i+1}' for i in user_indices]
        data = allocation[user_indices, :]
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=time_labels,
            y=user_labels,
            colorscale='Viridis',
            colorbar=dict(title='Bandwidth (Mbps)')
        ))
        
        fig.update_layout(
            title='Temporal Bandwidth Allocation Heatmap',
            xaxis_title='Time',
            yaxis_title='Users',
            height=600
        )
        
        return fig
    
    def plot_utilization_curve(self, utilization: np.ndarray, time_labels: List[str] = None) -> go.Figure:
       
        # Plot network utilization over time.
        # Args: utilization: Array of utilization values per time slot
        #       time_labels: Labels for time slots
        # Returns: Plotly figure
       
        if time_labels is None:
            time_labels = [f'{h:02d}:00' for h in range(len(utilization))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=utilization * 100,
            mode='lines+markers',
            name='Utilization',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add threshold lines
        fig.add_hline(y=90, line_dash="dash", line_color="red", 
                     annotation_text="High Load (90%)")
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                     annotation_text="Normal Load (50%)")
        
        fig.update_layout(
            title='Network Utilization Over Time',
            xaxis_title='Time',
            yaxis_title='Utilization (%)',
            height=400,
            yaxis_range=[0, 105]
        )
        
        return fig
    