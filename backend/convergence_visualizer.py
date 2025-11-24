# Real-time convergence plots showing optimization progress
import plotly.graph_objects as go
import numpy as np
from typing import Dict

class ConvergenceVisualizer:
    # Create convergence visualization plots.
    
    @staticmethod
    def create_objective_convergence_plot(convergence_data: Dict) -> go.Figure:
    
        #Creates focused objective convergence plot with annotations with Args: convergence_data: Dictionary with convergence tracking data and Returns: Plotly figure showing objective convergence
    
        iterations = convergence_data.get('iterations', [])
        obj_values = convergence_data.get('objective_values', [])
        
        if not iterations:
            return go.Figure().add_annotation(
                text="No convergence data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Main convergence line
        fig.add_trace(go.Scatter(
            x=iterations,
            y=obj_values,
            mode='lines+markers',
            name='Objective Value',
            line=dict(color='#1f77b4', width=4),
            marker=dict(size=8, color='#1f77b4', line=dict(width=2, color='white')),
            hovertemplate='<b>Iteration %{x}</b><br>Objective: %{y:.6f}<extra></extra>'
        ))
        
        # Add convergence rate annotation
        if len(obj_values) > 1:
            # Calculate improvement
            initial_obj = obj_values[0]
            final_obj = obj_values[-1]
            improvement = ((final_obj - initial_obj) / abs(initial_obj)) * 100 if initial_obj != 0 else 0
            
            # Add annotation
            fig.add_annotation(
                x=len(iterations) * 0.7,
                y=max(obj_values) * 0.9,
                text=f"<b>Total Improvement</b><br>{improvement:+.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#ff7f0e",
                ax=40,
                ay=-40,
                bordercolor="#ff7f0e",
                borderwidth=2,
                borderpad=4,
                bgcolor="white",
                font=dict(size=14, color="#ff7f0e")
            )
            
            # Mark initial and final points
            fig.add_trace(go.Scatter(
                x=[iterations[0], iterations[-1]],
                y=[obj_values[0], obj_values[-1]],
                mode='markers',
                name='Start/End',
                marker=dict(size=15, color=['green', 'red'], 
                          symbol=['circle', 'star'],
                          line=dict(width=2, color='white')),
                hovertemplate='<b>%{text}</b><br>Objective: %{y:.6f}<extra></extra>',
                text=['Start', 'End']
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '<b> Objective Function Convergence</b><br><sub>Watching the optimizer find the optimal solution</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=24, family='Arial Black')
            },
            xaxis_title='<b>Iteration</b>',
            yaxis_title='<b>Objective Value</b>',
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.5)',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
