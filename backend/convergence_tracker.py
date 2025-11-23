# Real-time convergence plots showing optimization progress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List

class ConvergenceVisualizer:
    # Create beautiful convergence visualization plots.
    
    @staticmethod
    def create_convergence_dashboard(convergence_data: Dict) -> go.Figure:
       
        # Create comprehensive convergence dashboard with Args: convergence_data: Dictionary with convergence tracking data and Returns: Plotly figure with 4 subplots showing convergence
       
        iterations = convergence_data.get('iterations', [])
        obj_values = convergence_data.get('objective_values', [])
        primal_res = convergence_data.get('primal_residuals', [])
        dual_res = convergence_data.get('dual_residuals', [])
        gaps = convergence_data.get('gaps', [])
        timestamps = convergence_data.get('timestamps', [])
        
        if not iterations:
            # Return empty figure if no data
            return go.Figure().add_annotation(
                text="No convergence data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Objective Value Convergence',
                'Optimality Gap',
                'Primal-Dual Residuals',
                'Time vs Objective'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Objective Value Convergence
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=obj_values,
                mode='lines+markers',
                name='Objective Value',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6),
                hovertemplate='Iteration: %{x}<br>Objective: %{y:.6f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Optimality Gap
        if gaps:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=gaps,
                    mode='lines+markers',
                    name='Optimality Gap',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6),
                    fill='tozeroy',
                    hovertemplate='Iteration: %{x}<br>Gap: %{y:.6e}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Primal-Dual Residuals
        if primal_res:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=primal_res,
                    mode='lines',
                    name='Primal Residual',
                    line=dict(color='#2ca02c', width=2),
                    hovertemplate='Iteration: %{x}<br>Primal Res: %{y:.6e}<extra></extra>'
                ),
                row=2, col=1
            )
        
        if dual_res:
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=dual_res,
                    mode='lines',
                    name='Dual Residual',
                    line=dict(color='#d62728', width=2),
                    hovertemplate='Iteration: %{x}<br>Dual Res: %{y:.6e}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Time vs Objective
        if timestamps:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=obj_values,
                    mode='lines+markers',
                    name='Time Progress',
                    line=dict(color='#9467bd', width=3),
                    marker=dict(size=6),
                    hovertemplate='Time: %{x:.3f}s<br>Objective: %{y:.6f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update axes
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
        
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)
        fig.update_yaxes(title_text="Gap", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Residual", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Objective Value", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="<b> Optimization Convergence Analysis</b>",
            title_x=0.5,
            title_font=dict(size=24, family='Arial Black'),
            hovermode='closest',
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        return fig
    
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
    
    @staticmethod
    def create_multi_objective_surface(fairness_vals: List[float],
                                      efficiency_vals: List[float],
                                      latency_vals: List[float]) -> go.Figure:
     
        # Creates 3D surface plot showing multi-objective tradeoffs.
        
        # Args:
        #     fairness_vals: List of fairness values
        #     efficiency_vals: List of efficiency values
        #     latency_vals: List of latency values
        
        # Returns:
        #     3D Plotly figure
    
        fig = go.Figure()
        
        # 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=fairness_vals,
            y=efficiency_vals,
            z=latency_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=latency_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Latency (ms)")
            ),
            text=[f'F:{f:.3f}, E:{e:.3f}, L:{l:.2f}' 
                  for f, e, l in zip(fairness_vals, efficiency_vals, latency_vals)],
            hovertemplate='<b>Multi-Objective Point</b><br>' +
                         'Fairness: %{x:.4f}<br>' +
                         'Efficiency: %{y:.4f}<br>' +
                         'Latency: %{z:.2f} ms<extra></extra>'
        ))
        
        fig.update_layout(
            title='<b> Multi-Objective Optimization Space</b>',
            scene=dict(
                xaxis_title='Fairness Score',
                yaxis_title='Efficiency Score',
                zaxis_title='Latency (ms)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700
        )
        
        return fig
    
    @staticmethod
    def create_convergence_animation(convergence_data: Dict) -> go.Figure:
    
        # Create animated convergence plot showing optimization progress.
        
        # Args:
        #     convergence_data: Dictionary with convergence tracking data
        
        # Returns:
        #     Animated Plotly figure
    
        iterations = convergence_data.get('iterations', [])
        obj_values = convergence_data.get('objective_values', [])
        
        if not iterations or len(iterations) < 2:
            return go.Figure().add_annotation(
                text="Insufficient data for animation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create frames for animation
        frames = []
        for i in range(1, len(iterations) + 1):
            frame_data = go.Scatter(
                x=iterations[:i],
                y=obj_values[:i],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4')
            )
            frames.append(go.Frame(data=[frame_data], name=str(i)))
        
        # Initial figure
        fig = go.Figure(
            data=[go.Scatter(
                x=iterations[:1],
                y=obj_values[:1],
                mode='markers',
                marker=dict(size=10, color='green')
            )],
            frames=frames
        )
        
        # Add play/pause buttons
        fig.update_layout(
            title='<b> Convergence Animation</b>',
            xaxis_title='Iteration',
            yaxis_title='Objective Value',
            height=600,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'y': 1.15
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }],
                        'label': str(i),
                        'method': 'animate'
                    }
                    for i, f in enumerate(frames)
                ],
                'x': 0.1,
                'len': 0.9,
                'y': 0,
                'pad': {'b': 10, 't': 50}
            }]
        )
        
        return fig
    
    @staticmethod
    def create_constraint_satisfaction_plot(result: Dict) -> go.Figure:
        """
        Create plot showing constraint satisfaction.
        
        Args:
            result: Optimization result dictionary
        
        Returns:
            Plotly figure
        """
        metrics = result.get('metrics', {})
        
        # Constraint satisfaction percentages
        constraints = [
            'Min Guarantee Met',
            'Max Limit Compliance',
            'Capacity Utilization',
            'User Satisfaction'
        ]
        
        percentages = [
            metrics.get('min_guarantee_met_percent', 0),
            metrics.get('max_compliance_percent', 0),
            result.get('capacity_utilization', 0) * 100,
            metrics.get('avg_satisfaction', 0) * 100
        ]
        
        colors = ['#2ca02c' if p >= 90 else '#ff7f0e' if p >= 70 else '#d62728' 
                 for p in percentages]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=constraints,
            y=percentages,
            marker=dict(color=colors, line=dict(color='black', width=2)),
            text=[f'{p:.1f}%' for p in percentages],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Satisfaction: %{y:.1f}%<extra></extra>'
        ))
        
        # Add target line at 100%
        fig.add_hline(y=100, line_dash="dash", line_color="green", 
                     annotation_text="Target: 100%")
        
        fig.update_layout(
            title='<b> Constraint Satisfaction Analysis</b>',
            yaxis_title='Satisfaction (%)',
            height=500,
            yaxis=dict(range=[0, 110]),
            showlegend=False
        )
        
        return fig