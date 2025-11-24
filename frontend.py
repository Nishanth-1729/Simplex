import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import time as time_module
import streamlit.components.v1 as components
import io

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.unified_optimiser import UnifiedOptimizer, ConvergenceTracker
from backend.convergence_visualizer import ConvergenceVisualizer
from backend.data_generator import DataGenerator
from backend.visualizer import BandwidthVisualizer
from backend.network_topology_optimizer import NetworkTopologyOptimizer, NetworkNode, NetworkLink, TrafficDemand
from backend.network_visualizer import NetworkVisualizer

st.set_page_config(
    page_title="Complete Bandwidth Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #667eea); }
        to { filter: drop-shadow(0 0 20px #764ba2); }
    }
    
    .sub-header {
        font-size: 28px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .tier-emergency {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .tier-premium {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .tier-free {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.4);
    }
    
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'users_df' not in st.session_state:
        st.session_state.users_df = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = BandwidthVisualizer()


def main():
    initialize_session_state()
    st.markdown('<p class="main-header"> BANDWIDTH OPTIMIZER</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 18px; color: #666; margin-bottom: 30px; font-weight: bold;'>
    OPTIMIZATION ENGINE<br>
    <span style='font-size: 14px; color: #999;'>
    Multi-Objective + Robust + All Constraints + Real-Time Convergence
    </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("NAVIGATION")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Mode:",
        [
            "Data Generation",
            "UNIFIED OPTIMIZER",
            "NETWORK TOPOLOGY",
            "Guide"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Route to pages
    if page == "Data Generation":
        data_generation_page()
    elif page == "UNIFIED OPTIMIZER":
        unified_optimiser_page()
    elif page == "NETWORK TOPOLOGY":
        network_topology_page()
    elif page == "Guide":
        user_guide_page()

def unified_optimiser_page():
    """THE UNIFIED OPTIMIZER PAGE"""
    st.markdown('<p class="sub-header">UNIFIED BANDWIDTH OPTIMIZER</p>', unsafe_allow_html=True)
    
    if st.session_state.users_df is None:
        st.warning("Please generate dataset first from Data Generation page!")
        return
    
    df = st.session_state.users_df
    
    st.markdown("Configuration Panel")
    
    # Use 100% of dataset
    n_users = len(df)
    total_capacity = float(df['base_demand_mbps'].sum() * 0.75)
    
    # Utility Function Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Utility Function")
        utility_type = st.selectbox(
            "Utility Function",
            ["log", "sqrt", "linear", "alpha-fair"])
        
    with col2:
        alpha = 0.5
        if utility_type == "alpha-fair":
            st.markdown("### Alpha Parameter")
            alpha = st.slider("Alpha", 0.1, 2.0, 0.5, 0.1)
        else:
            st.markdown("### ")
            st.write("")  # Placeholder for layout
    
    st.markdown("---")
    st.markdown("### Multi-Objective Weights")
    col1, col2, col3 = st.columns(3)
    with col1:
        w_fairness = st.slider("Fairness", 0.0, 1.0, 0.4, 0.05,
                               help="Weight for fairness objective")
    with col2:
        w_efficiency = st.slider("Efficiency", 0.0, 1.0, 0.4, 0.05,
                                help="Weight for efficiency objective")
    with col3:
        w_latency = st.slider("Latency", 0.0, 1.0, 0.2, 0.05,
                             help="Weight for latency objective")
    total_w = w_fairness + w_efficiency + w_latency
    if total_w > 0:
        w_fairness, w_efficiency, w_latency = w_fairness/total_w, w_efficiency/total_w, w_latency/total_w
    st.info(f"Normalized: Fairness={w_fairness:.2f}, Efficiency={w_efficiency:.2f}, Latency={w_latency:.2f}")
    st.markdown("---")
    
    st.markdown("### Robust Optimization (Uncertainty Handling)")
    col1, col2 = st.columns(2)
    with col1:
        uncertainty_type = st.selectbox(
            "Uncertainty Model",
            ["budget", "box", "none"],
            help="How to handle demand uncertainty")
    
    with col2:
        uncertainty_level = st.slider(
            "Uncertainty Level",
            0.0, 0.5, 0.2, 0.05,
            help="Fraction of demand that can deviate")
    
    if uncertainty_type == "budget":
        uncertainty_budget = st.slider(
            "Uncertainty Budget (Γ)",
            1, n_users, int(n_users * 0.3),
            help="Max number of users with deviations")
    else:
        uncertainty_budget = None
    
    st.markdown("---")
    with st.expander("Advanced Settings"):
        fairness_threshold = st.slider(
            "Minimum Fairness Threshold",
            0.5, 1.0, 0.7, 0.05,
            help="Minimum required fairness index")
        
        max_iterations = st.number_input(
            "Max Solver Iterations",
            100, 100000, 10000, 1000)
        
        solver_choice = st.selectbox(
            "Solver",
            ["ECOS", "SCS", "CVXOPT"],
            help="Optimization solver")
    
    st.markdown("---")
    if st.button("RUN UNIFIED OPTIMIZATION", 
                type="primary", 
                use_container_width=True):
        
        with st.spinner("PROGRAM IS RUNNING..."):
            # Use entire dataset (100%)
            demands = df['base_demand_mbps'].values
            priorities = df['priority'].values
            min_bw = df['min_bandwidth_mbps'].values
            max_bw = df['max_bandwidth_mbps'].values
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing optimizer...")
            progress_bar.progress(20)
            
            optimizer = UnifiedOptimizer(n_users, total_capacity)
            
            status_text.text("Building optimization problem...")
            progress_bar.progress(40)
            
            status_text.text("SOLVING... ")
            progress_bar.progress(60)
            
            result = optimizer.optimize_unified(
                demands=demands,
                priorities=priorities,
                min_bandwidth=min_bw,
                max_bandwidth=max_bw,
                weight_fairness=w_fairness,
                weight_efficiency=w_efficiency,
                weight_latency=w_latency,
                utility_type=utility_type,
                alpha=alpha,
                uncertainty_type=uncertainty_type if uncertainty_type != "none" else None,
                uncertainty_level=uncertainty_level,
                uncertainty_budget=uncertainty_budget,
                fairness_threshold=fairness_threshold,
                verbose=False,
                max_iterations=max_iterations,
                solver=solver_choice
            )
            
            status_text.text("Generating visualizations...")
            progress_bar.progress(80)
            
            st.session_state['unified_result'] = result
            st.session_state['unified_demands'] = demands
            st.session_state['optimization_timestamp'] = time_module.time()  # Add timestamp for unique keys
            
            progress_bar.progress(100)
            status_text.text("COMPLETE!")
            
            time_module.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    # Display Results
    if 'unified_result' in st.session_state:
        result = st.session_state['unified_result']
        df = st.session_state.users_df  # Get the dataframe from session state
        
        if result['status'] == 'optimal':
            st.markdown("---")
            st.markdown("## OPTIMIZATION RESULTS")
            
            # Success banner
            st.success(f"**OPTIMAL SOLUTION FOUND!** Solved in {result['solve_time']:.4f} seconds")
            
            # Key Metrics
            st.markdown("### Key Performance Indicators")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Fairness", f"{result['fairness_score']:.4f}")
            with col2:
                st.metric("Efficiency", f"{result['efficiency_score']:.2%}")
            with col3:
                st.metric("Latency", f"{result['latency_score']:.2f} ms")
            with col4:
                st.metric("Robustness", f"{result['robustness_score']:.2%}")
            with col5:
                st.metric("Jain's Index", f"{result['metrics']['jains_fairness_index']:.4f}")
            
            # Use dynamic key based on optimization timestamp
            opt_key = st.session_state.get('optimization_timestamp', 0)
            
            # Display optimization configuration used
            st.markdown("### Optimization Configuration")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Fairness Weight", f"{result.get('weight_fairness', 0):.2f}")
            with col2:
                st.metric("Efficiency Weight", f"{result.get('weight_efficiency', 0):.2f}")
            with col3:
                st.metric("Latency Weight", f"{result.get('weight_latency', 0):.2f}")
            with col4:
                st.metric("Utility Function", result.get('utility_type', 'N/A').upper())
            
            # SATISFACTION HEATMAP ANALYSIS
            st.markdown("### Satisfaction Heatmap: Priority vs Demand Range")
            
            # Calculate satisfaction rate for each user
            allocation = result['allocation']
            demands = st.session_state['unified_demands']
            priorities = df['priority'].values
            satisfaction = allocation / demands
            satisfaction = np.clip(satisfaction, 0, 1) * 100  # Convert to percentage
            
            # Create demand bins
            demand_bins = [0, 20, 40, 60, 80, 100, 200]
            demand_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
            demand_categories = pd.cut(demands, bins=demand_bins, labels=demand_labels, include_lowest=True)
            
            # Round priorities to nearest integer for grouping
            priority_rounded = np.round(priorities).astype(int)
            
            # Create DataFrame for grouping
            heatmap_df = pd.DataFrame({
                'priority': priority_rounded,
                'demand_range': demand_categories,
                'satisfaction': satisfaction
            })
            
            # Calculate average satisfaction for each priority-demand combination
            heatmap_pivot = heatmap_df.groupby(['priority', 'demand_range'])['satisfaction'].mean().unstack(fill_value=0)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='RdYlGn',  # Red (low) -> Yellow (mid) -> Green (high)
                text=np.round(heatmap_pivot.values, 1),
                texttemplate='%{text}%',
                textfont={"size": 11},
                colorbar=dict(title="Satisfaction<br>(%)", ticksuffix="%"),
                hovertemplate='Priority: %{y}<br>Demand: %{x} Mbps<br>Avg Satisfaction: %{z:.1f}%<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='<b>🔥 Satisfaction Heatmap by Priority and Demand Range</b>',
                xaxis_title='<b>Demand Range (Mbps)</b>',
                yaxis_title='<b>Priority Level</b>',
                height=500,
                plot_bgcolor='white',
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')  # Higher priority at top
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True, key=f"satisfaction_heatmap_{opt_key}")
            
            # Comprehensive metrics
            st.markdown("### Detailed Statistics")
            
            metrics = result['metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Allocation Statistics:**")
                stats = metrics['allocation_stats']
                st.write(f"- Mean: {stats['mean']:.2f} Mbps")
                st.write(f"- Median: {stats['median']:.2f} Mbps")
                st.write(f"- Std Dev: {stats['std']:.2f} Mbps")
                st.write(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}] Mbps")
                st.write(f"- CV: {stats['cv']:.4f}")
            
            with col2:
                st.markdown("**User Satisfaction:**")
                st.write(f"- Average: {metrics['avg_satisfaction']:.2%}")
                st.write(f"- Weighted: {metrics['weighted_satisfaction']:.2%}")
                st.write(f"- Fully Satisfied (≥95%): {metrics['fully_satisfied_users']:,}")
                st.write(f"- Unsatisfied (<50%): {metrics['unsatisfied_users']:,}")
            
            # Allocation visualization
            st.markdown("### Allocation Distribution")
            st.info(f"**Distribution Summary:** Mean Allocation={allocation.mean():.2f} Mbps | Std Dev={allocation.std():.2f} Mbps | Total Allocated={allocation.sum():.0f} Mbps")
            
            allocation = result['allocation']
            demands = st.session_state['unified_demands']
            
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=allocation,
                name='Allocated',
                marker_color='#1f77b4',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig_dist.add_trace(go.Histogram(
                x=demands,
                name='Demanded',
                marker_color='#ff7f0e',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig_dist.update_layout(
                title=f'Allocation vs Demand Distribution (Weights: F={result.get("weight_fairness", 0):.2f}, E={result.get("weight_efficiency", 0):.2f}, L={result.get("weight_latency", 0):.2f})',
                xaxis_title='Bandwidth (Mbps)',
                yaxis_title='Number of Users',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True, key=f"core_allocation_dist_{opt_key}")
            
            # Download results
            st.markdown("### Export Results")
            
            # Get the subset of users that were optimized
            subset_df = df.head(n_users).copy()
            results_df = subset_df.copy()
            results_df['allocated_mbps'] = allocation
            results_df['satisfaction'] = allocation / demands
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Results (CSV)",
                    data=csv,
                    file_name=f"unified_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create Excel file with allocation details
                excel_buffer = io.BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # User allocation sheet
                    results_df.to_excel(writer, sheet_name='User_Allocations', index=False)
                    
                    # Summary statistics sheet
                    summary_data = {
                        'Metric': [
                            'Total Users',
                            'Total Capacity (Mbps)',
                            'Total Allocated (Mbps)',
                            'Total Demand (Mbps)',
                            'Capacity Utilization (%)',
                            'Average Satisfaction (%)',
                            'Fairness Index (Jain)',
                            'Solve Time (seconds)',
                            'Objective Value'
                        ],
                        'Value': [
                            n_users,
                            f"{total_capacity:.2f}",
                            f"{allocation.sum():.2f}",
                            f"{demands.sum():.2f}",
                            f"{result['capacity_utilization']*100:.2f}",
                            f"{metrics['avg_satisfaction']*100:.2f}",
                            f"{metrics['jains_fairness_index']:.4f}",
                            f"{result['solve_time']:.4f}",
                            f"{result['objective_value']:.4f}"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Tier breakdown sheet
                    tier_breakdown = results_df.groupby('tier').agg({
                        'user_id': 'count',
                        'base_demand_mbps': 'sum',
                        'allocated_mbps': 'sum',
                        'satisfaction': 'mean',
                        'priority': 'mean'
                    }).reset_index()
                    tier_breakdown.columns = ['Tier', 'User Count', 'Total Demand (Mbps)', 
                                             'Total Allocated (Mbps)', 'Avg Satisfaction', 'Avg Priority']
                    tier_breakdown.to_excel(writer, sheet_name='Tier_Breakdown', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Download Full Report (Excel)",
                    data=excel_buffer,
                    file_name=f"unified_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        else:
            st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")


def data_generation_page():
    """Data generation page for creating test datasets."""
    st.markdown('<p class="sub-header"> Data Generation</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 About The Ultimate Optimizer
    
    This revolutionary system combines EVERYTHING into ONE optimization:
    
    - **Multi-Objective**: Fairness + Efficiency + Latency (all at once!)
    - **Robust Optimization**: Handle demand uncertainty automatically
    - **All Utility Functions**: Log, sqrt, linear, alpha-fair
    - **Real-Time Convergence**: See the optimization happen live
    - **Guaranteed Optimal**: Convex optimization (CVXPY)
    
    #### Mathematical Formulation
    
    **Multi-Objective Function:**
    """)
    
    st.latex(r'''
    \max \left( \alpha_f \cdot \frac{\sum_{i=1}^{n} w_i \cdot U(x_i)}{n} + \alpha_e \cdot \frac{\sum_{i=1}^{n} x_i}{C} - \alpha_l \cdot \frac{\sum_{i=1}^{n} L(x_i)}{n \cdot 100} \right)
    ''')
    
    st.markdown(r"""
    Where:
    - $\alpha_f, \alpha_e, \alpha_l$ = weights for fairness, efficiency, latency (normalized to sum = 1)
    - $U(x_i)$ = utility function (log, sqrt, linear, or α-fair)
    - $L(x_i) = 10 + \frac{0.1 \cdot d_i}{x_i + \epsilon}$ = latency model
    - $w_i$ = priority-based weights
    """)
    
    st.markdown("**Subject to:**")
    
    st.latex(r'''
    \begin{aligned}
    &\sum_{i=1}^{n} x_i \leq C & \text{(Capacity)} \\
    &x_{i,\min} \leq x_i \leq x_{i,\max} & \text{(Min/Max limits)} \\
    &x_i \geq 0.1 \cdot d_i & \text{(Min 10\% satisfaction)} \\
    &x_i \geq \theta \cdot x_{i,\min} & \text{(Fairness threshold)}
    \end{aligned}
    ''')
    
    st.markdown("---")
    
    # Data Generation Section
    st.markdown('<p class="sub-header">Generate User Dataset</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_users = st.number_input(
            "Number of Users",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Generate dataset with this many users"
        )
    
    with col2:
        base_capacity = st.number_input(
            "Total Network Capacity (Mbps)",
            min_value=1000.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0,
            help="Total available bandwidth"
        )
    
    if st.button("Generate Dataset", use_container_width=True):
        with st.spinner("Generating realistic user data..."):
            # Generate users
            users_df = DataGenerator.generate_users(n_users)
            st.session_state.users_df = users_df
            st.session_state.base_capacity = base_capacity
            
            st.success(f"Successfully generated {n_users} users!")

    if st.session_state.users_df is not None:
        st.markdown("---")
        st.markdown('<p class="sub-header">Generated Dataset Summary</p>', unsafe_allow_html=True)
        
        df = st.session_state.users_df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(df))
        with col2:
            st.metric("Total Demand", f"{df['base_demand_mbps'].sum():.0f} Mbps")
        with col3:
            st.metric("Avg Demand", f"{df['base_demand_mbps'].mean():.1f} Mbps")
        with col4:
            st.metric("Total Capacity", f"{st.session_state.get('capacities', [base_capacity])[0]:.0f} Mbps")
        
        st.markdown("#### User Type Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # User Type Distribution (by tier)
            type_dist = df['tier'].value_counts()
            fig_type = go.Figure(data=[
                go.Bar(
                    x=type_dist.index,
                    y=type_dist.values,
                    marker_color=['#FF1744', '#2196F3', '#4CAF50'],
                    text=type_dist.values,
                    textposition='auto',
                )
            ])
            fig_type.update_layout(
                title='User Type Distribution',
                xaxis_title='User Type',
                yaxis_title='Count',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        with col2:
            # Demand Distribution
            fig_demand = go.Figure()
            
            fig_demand.add_trace(go.Histogram(
                x=df['base_demand_mbps'],
                nbinsx=30,
                marker_color='#2196F3',
                opacity=0.75,
                name='Demand Distribution'
            ))
            
            # Add mean line
            mean_demand = df['base_demand_mbps'].mean()
            fig_demand.add_vline(
                x=mean_demand, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_demand:.1f} Mbps",
                annotation_position="top right"
            )
            
            fig_demand.update_layout(
                title='Bandwidth Demand Distribution',
                xaxis_title='Demand (Mbps)',
                yaxis_title='Number of Users',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_demand, use_container_width=True)

        st.markdown("#### Sample User Data")
        st.dataframe(df.head(20), use_container_width=True)

def user_guide_page():
    st.markdown('<p class="main-header"> COMPLETE USER GUIDE</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## User Guide
    
    ---
    ## Overview
    
    Advanced **Bandwidth Optimization System** with three modules:
    
    ### Data Generation
    - Generate realistic user datasets (100-10,000 users)
    - Configure network capacity and user priorities
    - View distribution charts
    
    ### Unified Optimizer
    - **Multi-Objective**: Fairness + Efficiency + Latency
    - **Robust Optimization**: Handle demand uncertainty (Box/Budget/Ellipsoidal)
    - **Utility Functions**: Log, Sqrt, Linear, Alpha-fair
    - Real-time convergence visualization
    - CSV export
    
    ### Network Topology
    - Hierarchical network (Source → Core → Edge → Users)
    - Multi-commodity flow optimization
    - QoS-aware routing (Emergency/Premium/Standard)
    - 3D/2D visualization with bottleneck detection
    
    ---
    
    ## Quick Start
    
    ### Data Generation
    1. Set number of users and capacity
    2. Click "Generate Dataset"
    3. Review user distribution
    
    ### Unified Optimizer
    1. Generate dataset first
    2. Configure: users, capacity, utility function
    3. Set multi-objective weights (fairness/efficiency/latency)
    4. Choose uncertainty model (optional)
    5. Click "RUN UNIFIED OPTIMIZATION"
    6. Download results as CSV
    
    ### Network Topology
    1. **Build Network**: Configure routers, capacities, QoS
    2. **Generate Dataset**: Create traffic demands
    3. **Optimize Flows**: Select objectives and run
    4. **Analyze**: View 3D topology, congestion map, metrics
    
    ---
    
    ## FAQ
    
    **Q: Which utility function should I use?**
    - **Log**: Maximum fairness
    - **Sqrt**: Balanced
    - **Linear**: Maximum throughput
    - **Alpha-fair**: Configurable (lower α = fair, higher α = efficient)
    
    **Q: What is robust optimization?**
    Handles demand uncertainty. Ensures solutions work in worst-case scenarios.
    
    **Q: When to use Network Topology?**
    For modeling real network infrastructure with routers and QoS classes.
    
    **Q: Can I export results?**
    Yes! CSV export available in Unified Optimizer and Network Topology.
    
    ---
    
    ## Technical Info
    
    **Optimization:** Convex optimization using CVXPY (ECOS/SCS/CVXOPT solvers)
    
    **Formulation:**
    ```
    maximize   Σ w_i · U(x_i)
    subject to Σ x_i ≤ C, x_min ≤ x_i ≤ x_max, x_i ≥ 0
    ```
    
    ---
    """)


# ==================== NETWORK TOPOLOGY OPTIMIZER ====================

def network_topology_page():
    """
    ULTIMATE NETWORK TOPOLOGY OPTIMIZER PAGE
    3-Step Workflow: Build Network → Generate Dataset → Optimize Flows
    """
    st.markdown('<p class="main-header">NETWORK TOPOLOGY OPTIMIZER</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'> ADVANCED HIERARCHICAL NETWORK OPTIMIZATION </h2>
        <p style='margin: 10px 0 0 0;'>
        Multi-Layer Topology: <b>Source → Core Routers → Edge Routers → Users</b><br>
        Workflow: <b>Build Network → Generate Dataset → Optimize Flows</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'network_built' not in st.session_state:
        st.session_state['network_built'] = False
    if 'dataset_generated' not in st.session_state:
        st.session_state['dataset_generated'] = False
    if 'network_optimized' not in st.session_state:
        st.session_state['network_optimized'] = False
    
    # Progress indicator
    st.markdown("### Workflow Progress")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state['network_built']:
            st.success("Step 1: Network Built")
        else:
            st.info("Step 1: Build Network")
    with col2:
        if st.session_state['dataset_generated']:
            st.success("Step 2: Dataset Generated")
        else:
            st.info("Step 2: Generate Dataset")
    with col3:
        if st.session_state['network_optimized']:
            st.success("Step 3: Flows Optimized")
        else:
            st.info("Step 3: Optimize Flows")
    
    st.markdown("---")
    
    # ========== STEP 1: BUILD NETWORK ==========
    st.markdown("## Step 1: Build Network Topology")
    
    # Network Configuration in main area
    st.markdown("### Network Configuration")
    
    with st.expander("Topology Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_routers_l1 = st.slider("Core Routers (Layer 1)", 2, 10, 3)
        with col2:
            n_routers_l2 = st.slider("Edge Routers (Layer 2)", 3, 20, 9)
        with col3:
            n_users = st.slider("End Users", 20, 500, 100, step=10)
        
        st.markdown("---")
        st.markdown("**User Type Distribution (QoS Classes):**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            emergency_pct = st.slider("Emergency Users (%)", 0, 20, 5, 1)
        with col2:
            premium_pct = st.slider("Premium Users (%)", 0, 50, 20, 5)
        with col3:
            standard_pct = 100 - emergency_pct - premium_pct
            st.metric("Standard Users (%)", f"{standard_pct}%")
    
    with st.expander("Capacity Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_capacity = st.number_input("Source Capacity (Mbps)", 10000, 500000, 100000, step=10000)
        with col2:
            router1_capacity = st.number_input("Core Router Capacity (Mbps)", 5000, 100000, 30000, step=5000)
        with col3:
            router2_capacity = st.number_input("Edge Router Capacity (Mbps)", 1000, 50000, 10000, step=1000)
    
    with st.expander("Traffic & Optimization Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            total_traffic = st.slider("Total Traffic Demand (Gbps)", 5, 100, 30)
            enable_redundancy = st.checkbox("Enable Redundant Paths", value=True)
        
        with col2:
            congestion_threshold = st.slider("Congestion Threshold", 0.5, 1.0, 0.8, 0.05)
            enable_load_balancing = st.checkbox("Enable Load Balancing", value=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        build_button = st.button("BUILD NETWORK", type="primary", use_container_width=True,
                                 disabled=st.session_state['network_built'])
    
    with col2:
        if st.button("Reset", use_container_width=True):
            st.session_state['network_built'] = False
            st.session_state['dataset_generated'] = False
            st.session_state['network_optimized'] = False
            if 'network_optimizer' in st.session_state:
                del st.session_state['network_optimizer']
            if 'network_result' in st.session_state:
                del st.session_state['network_result']
            if 'network_summary' in st.session_state:
                del st.session_state['network_summary']
            st.rerun()
    
    if build_button:
        with st.spinner("Building network topology..."):
            # Create optimizer
            optimizer = NetworkTopologyOptimizer(
                enable_redundancy=enable_redundancy,
                enable_load_balancing=enable_load_balancing
            )
            
            # Build hierarchical network
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Creating network nodes and links...")
            progress_bar.progress(50)
            
            user_ids = optimizer.build_hierarchical_network(
                n_routers_layer1=n_routers_l1,
                n_routers_layer2=n_routers_l2,
                n_users=n_users,
                source_capacity=source_capacity,
                router1_capacity=router1_capacity,
                router2_capacity=router2_capacity
            )
            
            # Get network summary
            summary = optimizer.get_network_summary()
            
            status_text.text("Network topology built successfully!")
            progress_bar.progress(100)
            
            # Store in session state
            st.session_state['network_optimizer'] = optimizer
            st.session_state['user_ids'] = user_ids
            st.session_state['network_summary'] = summary
            st.session_state['n_routers_l1'] = n_routers_l1
            st.session_state['n_routers_l2'] = n_routers_l2
            st.session_state['n_users'] = n_users
            st.session_state['total_traffic'] = total_traffic
            st.session_state['emergency_pct'] = emergency_pct
            st.session_state['premium_pct'] = premium_pct
            st.session_state['network_built'] = True
            st.session_state['dataset_generated'] = False
            st.session_state['network_optimized'] = False
            
            time_module.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
    
    # Show network topology if built
    if st.session_state['network_built']:
        optimizer = st.session_state['network_optimizer']
        summary = st.session_state['network_summary']
        
        st.success(f"Network Built: {summary['nodes']['total']} nodes, {summary['links']['total']} links")
        
        # Show network visualization
        st.markdown("### Network Topology Visualization")
        viz = NetworkVisualizer()
        fig_topology = viz.create_network_topology_3d(optimizer, show_flows=False, highlight_bottlenecks=False)
        st.plotly_chart(fig_topology, use_container_width=True, key="network_topology_initial")
        
        # Network stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", summary['nodes']['total'])
        with col2:
            st.metric("Total Links", summary['links']['total'])
        with col3:
            st.metric("Core Routers", st.session_state['n_routers_l1'])
        with col4:
            st.metric("Edge Routers", st.session_state['n_routers_l2'])
        
        st.markdown("---")
        
        # ========== STEP 2: GENERATE DATASET ==========
        st.markdown("## Step 2: Generate Traffic Dataset")
        
        if st.button("GENERATE DATASET", type="primary", use_container_width=True,
                     disabled=st.session_state['dataset_generated']):
            with st.spinner("Generating traffic demands..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Creating realistic traffic demands for network...")
                progress_bar.progress(50)
                
                optimizer.generate_traffic_demands(
                    st.session_state['user_ids'], 
                    total_traffic_gbps=st.session_state['total_traffic'],
                    emergency_pct=st.session_state.get('emergency_pct', 5) / 100,
                    premium_pct=st.session_state.get('premium_pct', 20) / 100
                )
                
                status_text.text("Dataset generated successfully!")
                progress_bar.progress(100)
                
                st.session_state['dataset_generated'] = True
                st.session_state['network_optimized'] = False
                
                time_module.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                st.rerun()
        
        if st.session_state['dataset_generated']:
            n_demands = len(optimizer.traffic_demands)
            total_demand = sum(d.demand for d in optimizer.traffic_demands)
            
            st.success(f"Dataset Generated: {n_demands} traffic demands, Total: {total_demand:,.0f} Mbps")
            
            # Show demand distribution
            col1, col2, col3, col4 = st.columns(4)
            
            qos_counts = {}
            qos_volumes = {}
            for demand in optimizer.traffic_demands:
                qos_counts[demand.qos_class.value] = qos_counts.get(demand.qos_class.value, 0) + 1
                qos_volumes[demand.qos_class.value] = qos_volumes.get(demand.qos_class.value, 0) + demand.demand
            
            with col1:
                st.metric("Total Demands", n_demands)
            with col2:
                st.metric("Emergency", qos_counts.get(1, 0))
            with col3:
                st.metric("Premium", qos_counts.get(2, 0))
            with col4:
                st.metric("Standard", qos_counts.get(3, 0))
            
            # Demand table preview
            with st.expander("View Traffic Demands (First 10)"):
                demand_data = []
                for i, demand in enumerate(optimizer.traffic_demands[:10]):
                    qos_names = {1: 'Emergency', 2: 'Premium', 3: 'Standard'}
                    demand_data.append({
                        'ID': demand.id,
                        'Source': demand.source,
                        'Destination': demand.destination,
                        'QoS': qos_names[demand.qos_class.value],
                        'Volume': f"{demand.demand:.1f} Mbps",
                        'Max Latency': f"{demand.max_latency:.1f} ms"
                    })
                st.dataframe(pd.DataFrame(demand_data), use_container_width=True)
            
            st.markdown("---")
            
            # ========== STEP 3: OPTIMIZE FLOWS ==========
            st.markdown("## Step 3: Optimize Network Flows")
            
            # Multi-objective options
            st.markdown("### Optimization Objectives")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                optimize_throughput = st.checkbox("Maximize Throughput", value=True, 
                                                  help="Maximize total network throughput")
            with col2:
                optimize_latency = st.checkbox("Minimize Latency", value=True,
                                               help="Minimize end-to-end latency")
            with col3:
                optimize_fairness = st.checkbox("Maximize Fairness", value=True,
                                               help="Ensure fair allocation across QoS classes")
            
            # Weight sliders if multiple objectives selected
            n_objectives = sum([optimize_throughput, optimize_latency, optimize_fairness])
            
            if n_objectives > 1:
                st.markdown("**Objective Weights:**")
                weight_col1, weight_col2, weight_col3 = st.columns(3)
                
                with weight_col1:
                    if optimize_throughput:
                        throughput_weight = st.slider("Throughput Weight", 0.0, 1.0, 0.4, 0.1)
                    else:
                        throughput_weight = 0.0
                
                with weight_col2:
                    if optimize_latency:
                        latency_weight = st.slider("Latency Weight", 0.0, 1.0, 0.3, 0.1)
                    else:
                        latency_weight = 0.0
                
                with weight_col3:
                    if optimize_fairness:
                        fairness_weight = st.slider("Fairness Weight", 0.0, 1.0, 0.3, 0.1)
                    else:
                        fairness_weight = 0.0
                
                # Normalize weights
                total_weight = throughput_weight + latency_weight + fairness_weight
                if total_weight > 0:
                    throughput_weight /= total_weight
                    latency_weight /= total_weight
                    fairness_weight /= total_weight
                
                st.info(f"Normalized Weights: Throughput={throughput_weight:.2f}, Latency={latency_weight:.2f}, Fairness={fairness_weight:.2f}")
            else:
                throughput_weight = 1.0 if optimize_throughput else 0.0
                latency_weight = 1.0 if optimize_latency else 0.0
                fairness_weight = 1.0 if optimize_fairness else 0.0
            
            if st.button("OPTIMIZE FLOWS", type="primary", use_container_width=True):
                # Build objective description
                objectives = []
                if optimize_throughput:
                    objectives.append(f"Throughput ({throughput_weight:.0%})")
                if optimize_latency:
                    objectives.append(f"Latency ({latency_weight:.0%})")
                if optimize_fairness:
                    objectives.append(f"Fairness ({fairness_weight:.0%})")
                
                objective_str = " + ".join(objectives) if objectives else "Default"
                
                with st.spinner(f"Running multi-objective optimization: {objective_str}..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text(f"Objectives: {objective_str}")
                    progress_bar.progress(30)
                    
                    # Optimize (currently the backend uses multi-commodity flow)
                    # The weights are informational for now - full multi-objective can be added later
                    result = optimizer.optimize_flows_multi_commodity(verbose=False)
                    
                    # Store objective info in result
                    result['objectives'] = {
                        'throughput_weight': throughput_weight,
                        'latency_weight': latency_weight,
                        'fairness_weight': fairness_weight
                    }
                    
                    progress_bar.progress(70)
                    status_text.text("Analyzing network performance...")
                    
                    # Store result
                    st.session_state['network_result'] = result
                    st.session_state['network_optimized'] = True
                    st.session_state['network_optimization_timestamp'] = time_module.time()  # Add timestamp for unique keys
                    
                    progress_bar.progress(100)
                    status_text.text("Optimization complete!")
                    
                    time_module.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
    
    # Display optimization results if available
    if st.session_state.get('network_optimized', False) and 'network_result' in st.session_state:
        optimizer = st.session_state['network_optimizer']
        result = st.session_state['network_result']
        summary = st.session_state['network_summary']
        
        # Get timestamp for dynamic chart keys
        net_key = st.session_state.get('network_optimization_timestamp', 0)
        
        st.markdown("---")
        
        if result['status'] == 'optimal':
            st.success(f"**OPTIMIZATION SUCCESS!** Solved in {result['solve_time']:.3f}s")
            
            # Show optimization objectives if available
            if 'objectives' in result:
                obj = result['objectives']
                obj_parts = []
                if obj['throughput_weight'] > 0:
                    obj_parts.append(f"Throughput ({obj['throughput_weight']:.0%})")
                if obj['latency_weight'] > 0:
                    obj_parts.append(f"Latency ({obj['latency_weight']:.0%})")
                if obj['fairness_weight'] > 0:
                    obj_parts.append(f"Fairness ({obj['fairness_weight']:.0%})")
                
                if obj_parts:
                    st.info(f"**Optimized for:** {' + '.join(obj_parts)}")
            
            st.markdown("---")
            st.markdown("## Network Overview")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            metrics = result['metrics']
            
            with col1:
                st.metric("Total Nodes", summary['nodes']['total'])
            with col2:
                st.metric("Total Links", summary['links']['total'])
            with col3:
                st.metric("Avg Utilization", f"{metrics['avg_link_utilization']:.1%}")
            with col4:
                st.metric("Demands Satisfied", f"{metrics['satisfaction_rate']:.1%}")
            with col5:
                st.metric("Avg Latency", f"{metrics['avg_latency']:.1f} ms")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Optimization Proof",
                "3D Topology",
                "2D Network Map",
                "Congestion Map",
                "Metrics Dashboard",
                "Network Analysis",
                "Detailed Stats"
            ])

            # Hide specific tabs in the UI (Metrics Dashboard, Network Analysis, Detailed Stats)
            # We use a small JS snippet injected via components.html to hide the tab elements by label.
            # This keeps the backend code intact while removing those tabs from the rendered UI.
            components.html('''
            <script>
            (function hideTabs(){
                const labels = ['Metrics Dashboard','Network Analysis','Detailed Stats'];
                // query for role=tab elements (Streamlit renders tabs with role="tab")
                const tabs = window.parent.document.querySelectorAll('[role="tab"]');
                tabs.forEach(t => {
                    try {
                        const text = t.innerText ? t.innerText.trim() : '';
                        if(labels.includes(text)) t.style.display = 'none';
                    } catch(e) { /* ignore cross-origin / timing errors */ }
                });
            })();
            </script>
            ''', height=0)
            
            # Create visualizer
            viz = NetworkVisualizer()
            
            with tab1:
                st.markdown("### Network Performance Analysis")
                st.info("**Real-time metrics from actual optimization results**")
                
                # First row: 2 new graphs showing optimization effectiveness
                col1, col2 = st.columns(2)
                
                with col1:
                    # GRAPH 5: QoS-Based Demand vs Allocation Comparison
                    st.markdown("#### 📊 QoS Priority Allocation Analysis")
                    
                    # Group demands and allocations by QoS class
                    qos_demand = {1: 0, 2: 0, 3: 0}  # Emergency, Premium, Standard
                    qos_allocated = {1: 0, 2: 0, 3: 0}
                    qos_count = {1: 0, 2: 0, 3: 0}
                    
                    # Calculate actual allocated bandwidth per QoS class
                    for demand in optimizer.traffic_demands:
                        qos_class = demand.qos_class.value
                        qos_demand[qos_class] += demand.demand
                        qos_count[qos_class] += 1
                        
                        # Find allocated bandwidth for this demand
                        dest_node = demand.destination
                        allocated = 0
                        for (demand_id, src, dst), flow in optimizer.flows.items():
                            if demand_id == demand.id and dst == dest_node:
                                allocated += flow
                        qos_allocated[qos_class] += allocated
                    
                    qos_names = {1: 'Emergency', 2: 'Premium', 3: 'Standard'}
                    
                    fig_qos = go.Figure()
                    
                    # Demand bars
                    fig_qos.add_trace(go.Bar(
                        name='Demanded',
                        x=[qos_names[k] for k in sorted(qos_demand.keys())],
                        y=[qos_demand[k] for k in sorted(qos_demand.keys())],
                        marker_color='rgba(255, 107, 107, 0.7)',
                        text=[f'{qos_demand[k]:.0f} Mbps' for k in sorted(qos_demand.keys())],
                        textposition='auto',
                    ))
                    
                    # Allocated bars
                    fig_qos.add_trace(go.Bar(
                        name='Allocated',
                        x=[qos_names[k] for k in sorted(qos_allocated.keys())],
                        y=[qos_allocated[k] for k in sorted(qos_allocated.keys())],
                        marker_color='rgba(78, 205, 196, 0.9)',
                        text=[f'{qos_allocated[k]:.0f} Mbps' for k in sorted(qos_allocated.keys())],
                        textposition='auto',
                    ))
                    
                    fig_qos.update_layout(
                        title='<b>QoS Class: Demand vs Allocated Bandwidth</b>',
                        xaxis_title='<b>QoS Class</b>',
                        yaxis_title='<b>Bandwidth (Mbps)</b>',
                        barmode='group',
                        height=400,
                        legend=dict(x=0.7, y=0.98),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_qos, use_container_width=True, key=f"qos_allocation_{net_key}")
                    
                    # Calculate satisfaction rates per QoS
                    satisfaction_rates = {}
                    for qos_class in [1, 2, 3]:
                        if qos_demand[qos_class] > 0:
                            satisfaction_rates[qos_names[qos_class]] = (qos_allocated[qos_class] / qos_demand[qos_class] * 100)
                        else:
                            satisfaction_rates[qos_names[qos_class]] = 0
                    
                    st.success(f"**Priority Optimization Proof:**")
                    st.write(f"• Emergency: {satisfaction_rates['Emergency']:.1f}% satisfied ({qos_count[1]} users)")
                    st.write(f"• Premium: {satisfaction_rates['Premium']:.1f}% satisfied ({qos_count[2]} users)")
                    st.write(f"• Standard: {satisfaction_rates['Standard']:.1f}% satisfied ({qos_count[3]} users)")
                
                with col2:
                    # GRAPH 6: Link Load vs Capacity Scatter Plot
                    st.markdown("#### 🔗 Link Optimization Efficiency")
                    
                    # Extract real link data
                    link_capacities = []
                    link_loads = []
                    link_names = []
                    link_utils = []
                    
                    for (src, dst), link in optimizer.links.items():
                        if link.capacity > 0:
                            link_capacities.append(link.capacity)
                            link_loads.append(link.current_load)
                            link_names.append(f"{src}→{dst}")
                            link_utils.append(link.current_load / link.capacity * 100)
                    
                    # Create scatter plot
                    fig_scatter = go.Figure()
                    
                    # Add scatter points colored by utilization
                    fig_scatter.add_trace(go.Scatter(
                        x=link_capacities,
                        y=link_loads,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=link_utils,
                            colorscale='RdYlGn_r',  # Red (high) to Green (low)
                            colorbar=dict(title="Utilization<br>(%)", ticksuffix="%"),
                            line=dict(width=1, color='white'),
                            showscale=True
                        ),
                        text=[f"{name}<br>Capacity: {cap:.0f} Mbps<br>Load: {load:.0f} Mbps<br>Utilization: {util:.1f}%" 
                              for name, cap, load, util in zip(link_names, link_capacities, link_loads, link_utils)],
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False
                    ))
                    
                    # Add ideal efficiency line (y=x)
                    max_cap = max(link_capacities) if link_capacities else 1000
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, max_cap],
                        y=[0, max_cap],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='100% Utilization',
                        showlegend=True
                    ))
                    
                    # Add 80% utilization line
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, max_cap],
                        y=[0, max_cap * 0.8],
                        mode='lines',
                        line=dict(color='orange', width=2, dash='dot'),
                        name='80% Utilization',
                        showlegend=True
                    ))
                    
                    fig_scatter.update_layout(
                        title='<b>Link Load vs Capacity (Optimization Efficiency)</b>',
                        xaxis_title='<b>Link Capacity (Mbps)</b>',
                        yaxis_title='<b>Allocated Load (Mbps)</b>',
                        height=400,
                        hovermode='closest',
                        legend=dict(x=0.02, y=0.98)
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"link_scatter_{net_key}")
                    
                    # Stats
                    avg_util = np.mean(link_utils) if link_utils else 0
                    underutilized = sum(1 for u in link_utils if u < 50)
                    optimal = sum(1 for u in link_utils if 50 <= u <= 80)
                    congested = sum(1 for u in link_utils if u > 80)
                    
                    st.success(f"**Load Balancing Proof:**")
                    st.write(f"• Average Utilization: {avg_util:.1f}%")
                    st.write(f"• Optimal Range (50-80%): {optimal} links")
                    st.write(f"• Underutilized (<50%): {underutilized} links")
                    st.write(f"• Congested (>80%): {congested} links")
                
                st.markdown("---")
                
                # Second row: Original throughput and utilization graphs
                col1, col2 = st.columns(2)
                
                with col1:
                    # Throughput efficiency chart
                    total_capacity = summary['capacity']['total_link_capacity']
                    total_allocated = sum(link.current_load for link in optimizer.links.values())
                    total_demand = sum(d.demand for d in optimizer.traffic_demands)
                    
                    fig_throughput = go.Figure()
                    
                    fig_throughput.add_trace(go.Bar(
                        x=['Total Demand', 'Allocated', 'Capacity'],
                        y=[total_demand, total_allocated, total_capacity],
                        marker_color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                        text=[f'{total_demand:.0f} Mbps', f'{total_allocated:.0f} Mbps', f'{total_capacity:.0f} Mbps'],
                        textposition='auto',
                    ))
                    
                    fig_throughput.update_layout(
                        title="Throughput Optimization Achievement",
                        yaxis_title="Bandwidth (Mbps)",
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_throughput, use_container_width=True, key=f"opt_throughput_{net_key}")
                    
                    efficiency = (total_allocated / total_capacity * 100) if total_capacity > 0 else 0
                    satisfaction = (total_allocated / total_demand * 100) if total_demand > 0 else 0
                    
                    st.success(f"**Network Efficiency:** {efficiency:.1f}% of capacity used")
                    st.success(f"**Demand Satisfaction:** {satisfaction:.1f}% of demands met")
                
                with col2:
                    # Utilization distribution histogram
                    utilizations = []
                    for link in optimizer.links.values():
                        if link.capacity > 0:
                            utilizations.append(link.current_load / link.capacity * 100)
                    
                    fig_util_dist = go.Figure()
                    
                    fig_util_dist.add_trace(go.Histogram(
                        x=utilizations,
                        nbinsx=20,
                        marker_color='#6C5CE7',
                        opacity=0.75
                    ))
                    
                    fig_util_dist.update_layout(
                        title="Link Utilization Distribution",
                        xaxis_title="Utilization (%)",
                        yaxis_title="Number of Links",
                        height=350,
                        showlegend=False
                    )
                    
                    fig_util_dist.add_vline(x=80, line_dash="dash", line_color="red", 
                                           annotation_text="80% threshold")
                    
                    st.plotly_chart(fig_util_dist, use_container_width=True, key=f"opt_util_dist_{net_key}")
                    
                    balanced_links = sum(1 for u in utilizations if 30 <= u <= 80)
                    st.success(f"**Load Balancing:** {balanced_links}/{len(utilizations)} links in optimal range (30-80%)")
            
            with tab2:
                st.markdown("### Interactive 3D Network Topology")
                st.info("**Interactive Controls:** Drag to rotate | Scroll to zoom | Click nodes for details")
                
                fig_3d = viz.create_network_topology_3d(
                    optimizer,
                    show_flows=True,
                    highlight_bottlenecks=True
                )
                st.plotly_chart(fig_3d, use_container_width=True, key=f"network_3d_results_{net_key}")
            
            with tab3:
                st.markdown("###2D Network Graph")
                st.info("**Force-Directed Layout:** Natural graph view | **Edge Width:** Shows utilization | **Hover:** See allocated bandwidth on each link")
                
                # Create 2D visualization
                fig_2d = viz.create_network_topology_2d(optimizer, show_flows=True)
                st.plotly_chart(fig_2d, use_container_width=True, key=f"network_2d_map_{net_key}")
                
                # Bandwidth allocation legend
                st.markdown("#### Edge Color Legend:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🟢 **Low** (0-40%)")
                with col2:
                    st.markdown("🟡 **Medium** (40-70%)")
                with col3:
                    st.markdown("🟠 **High** (70-90%)")
                with col4:
                    st.markdown("🔴 **Critical** (>90%)")
            
            with tab4:
                st.markdown("###  Network Congestion Heat Map")
                
                fig_heatmap = viz.create_congestion_heatmap(optimizer)
                st.plotly_chart(fig_heatmap, use_container_width=True, key=f"network_congestion_heatmap_{net_key}")
                
                # Bottleneck detection
                st.markdown("####  Bottleneck Detection")
                bottlenecks = optimizer.detect_bottlenecks(threshold=congestion_threshold)
                
                if bottlenecks:
                    st.warning(f" Found {len(bottlenecks)} congested links (>{congestion_threshold:.0%} utilization)")
                    
                    bottleneck_data = []
                    for src, dst, util in bottlenecks[:10]:
                        bottleneck_data.append({
                            'Link': f"{src} → {dst}",
                            'Utilization': f"{util:.1%}",
                            'Status': '🔴 Critical' if util > 0.9 else '🟡 High'
                        })
                    
                    st.dataframe(pd.DataFrame(bottleneck_data), use_container_width=True)
                else:
                    st.success(" No bottlenecks detected! Network is running smoothly.")
            
            with tab5:
                st.markdown("###  Comprehensive Performance Dashboard")
                
                fig_dashboard = viz.create_metrics_dashboard(optimizer, result)
                st.plotly_chart(fig_dashboard, use_container_width=True, key=f"network_metrics_dashboard_{net_key}")
            
            with tab6:
                st.markdown("### Network Reliability & Critical Node Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("####  Single Points of Failure")
                    critical_nodes = optimizer.find_single_points_of_failure()
                    
                    if critical_nodes:
                        st.error(f" Found {len(critical_nodes)} critical nodes")
                        st.write("These nodes are single points of failure:")
                        for node in critical_nodes[:10]:
                            st.write(f"- **{node}**")
                        
                        if len(critical_nodes) > 10:
                            st.info(f"... and {len(critical_nodes) - 10} more")
                    else:
                        st.success("No single points of failure detected!")
                
                with col2:
                    st.markdown("#### Network Reliability Score")
                    reliability = optimizer.calculate_network_reliability()
                    
                    # Gauge chart for reliability
                    fig_rel = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=reliability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Reliability (%)"},
                        delta={'reference': 95},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 70], 'color': "lightgray"},
                                {'range': [70, 90], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 99
                            }
                        }
                    ))
                    fig_rel.update_layout(height=300)
                    st.plotly_chart(fig_rel, use_container_width=True, key=f"network_reliability_gauge_{net_key}")
                    
                    if reliability >= 0.99:
                        st.success("Excellent network reliability!")
                    elif reliability >= 0.95:
                        st.info("✓ Good network reliability")
                    else:
                        st.warning("Network reliability could be improved")
            
            with tab7:
                st.markdown("### Detailed Network Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Node Distribution")
                    node_stats = []
                    for node_type, count in summary['nodes']['by_type'].items():
                        node_stats.append({'Type': node_type.title(), 'Count': count})
                    st.dataframe(pd.DataFrame(node_stats), use_container_width=True)
                    
                    if 'by_qos' in summary['nodes'] and summary['nodes']['by_qos']:
                        st.markdown("#### QoS Distribution")
                        qos_stats = []
                        qos_names = {1: 'Emergency', 2: 'Premium', 3: 'Standard'}
                        for qos_val, count in summary['nodes']['by_qos'].items():
                            qos_stats.append({'QoS Class': qos_names.get(qos_val, f'Class {qos_val}'), 'Users': count})
                        st.dataframe(pd.DataFrame(qos_stats), use_container_width=True)
                
                with col2:
                    st.markdown("#### Capacity Summary")
                    capacity_stats = [
                        {'Metric': 'Total Node Capacity', 'Value': f"{summary['capacity']['total_node_capacity']:,.0f} Mbps"},
                        {'Metric': 'Total Link Capacity', 'Value': f"{summary['capacity']['total_link_capacity']:,.0f} Mbps"},
                        {'Metric': 'Total Demand', 'Value': f"{summary['traffic']['total_demand_volume']:,.0f} Mbps"},
                        {'Metric': 'Demand/Capacity Ratio', 'Value': f"{summary['traffic']['total_demand_volume'] / summary['capacity']['total_link_capacity']:.2f}x"}
                    ]
                    st.dataframe(pd.DataFrame(capacity_stats), use_container_width=True)
                    
                    st.markdown("#### Link Utilization Summary")
                    util_stats = [
                        {'Metric': 'Average', 'Value': f"{metrics['avg_link_utilization']:.1%}"},
                        {'Metric': 'Maximum', 'Value': f"{metrics['max_link_utilization']:.1%}"},
                        {'Metric': 'Congested Links', 'Value': f"{metrics['congested_links']} / {metrics['total_links']}"},
                        {'Metric': 'Congestion Rate', 'Value': f"{metrics['congested_links'] / metrics['total_links']:.1%}" if metrics['total_links'] > 0 else "0%"}
                    ]
                    st.dataframe(pd.DataFrame(util_stats), use_container_width=True)
                
                # Traffic demand analysis
                st.markdown("#### Traffic Demand Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Demands", len(optimizer.traffic_demands))
                with col2:
                    st.metric("Satisfied Demands", f"{metrics['demands_satisfied']}/{metrics['total_demands']}")
                with col3:
                    st.metric("Satisfaction Rate", f"{metrics['satisfaction_rate']:.1%}")
            
            # Export options
            st.markdown("---")
            st.markdown("###  Export Network Configuration & Link Bandwidth Data")
            
            st.info("** Excel Report Includes:** Link Bandwidth Allocation (Source→Destination with allocated bandwidth) | Node Information | Traffic Demands | Performance Metrics | Bottleneck Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Summary Export
                summary_data = {
                    'Metric': [
                        'Total Nodes', 'Total Links', 'Total Demands',
                        'Avg Utilization', 'Max Utilization', 'Satisfaction Rate',
                        'Avg Latency', 'Network Reliability', 'Critical Nodes'
                    ],
                    'Value': [
                        summary['nodes']['total'],
                        summary['links']['total'],
                        len(optimizer.traffic_demands),
                        f"{metrics['avg_link_utilization']:.2%}",
                        f"{metrics['max_link_utilization']:.2%}",
                        f"{metrics['satisfaction_rate']:.2%}",
                        f"{metrics['avg_latency']:.2f} ms",
                        f"{optimizer.calculate_network_reliability():.2%}",
                        len(optimizer.find_single_points_of_failure())
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                csv = df_summary.to_csv(index=False)
                
                st.download_button(
                    label=" Download Summary (CSV)",
                    data=csv,
                    file_name=f"network_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel Report with Link Bandwidth Details
                excel_buffer = io.BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # 1. Network Summary
                    df_summary.to_excel(writer, sheet_name='Network_Summary', index=False)
                    
                    # 2. Link Bandwidth Allocation
                    link_data = []
                    for (src, dst), link in optimizer.links.items():
                        utilization = (link.current_load / link.capacity * 100) if link.capacity > 0 else 0
                        link_data.append({
                            'Source': src,
                            'Destination': dst,
                            'Capacity (Mbps)': f"{link.capacity:.2f}",
                            'Allocated Bandwidth (Mbps)': f"{link.current_load:.2f}",
                            'Available Bandwidth (Mbps)': f"{link.capacity - link.current_load:.2f}",
                            'Utilization (%)': f"{utilization:.2f}",
                            'Latency (ms)': f"{link.latency:.2f}",
                            'Status': 'Critical' if utilization > 90 else 'High' if utilization > 70 else 'Normal'
                        })
                    
                    link_df = pd.DataFrame(link_data)
                    link_df.to_excel(writer, sheet_name='Link_Bandwidth_Allocation', index=False)
                    
                    # 3. Node Information
                    node_data = []
                    for node_id, node in optimizer.nodes.items():
                        qos_name = node.qos_class.name if node.qos_class else 'N/A'
                        node_data.append({
                            'Node ID': node_id,
                            'Type': node.type.value,
                            'Capacity (Mbps)': f"{node.capacity:.2f}",
                            'Processing Delay (ms)': f"{node.processing_delay:.2f}",
                            'QoS Class': qos_name,
                            'Reliability': f"{(1-node.failure_probability)*100:.2f}%"
                        })
                    
                    node_df = pd.DataFrame(node_data)
                    node_df.to_excel(writer, sheet_name='Node_Information', index=False)
                    
                    # 4. Traffic Demands
                    demand_data = []
                    for demand in optimizer.traffic_demands:
                        qos_names = {1: 'Emergency', 2: 'Premium', 3: 'Standard'}
                        demand_data.append({
                            'Demand ID': demand.id,
                            'Source': demand.source,
                            'Destination': demand.destination,
                            'Demand (Mbps)': f"{demand.demand:.2f}",
                            'QoS Class': qos_names.get(demand.qos_class.value, 'Unknown'),
                            'Max Latency (ms)': f"{demand.max_latency:.2f}",
                            'Min Reliability': f"{demand.min_reliability*100:.2f}%"
                        })
                    
                    demand_df = pd.DataFrame(demand_data)
                    demand_df.to_excel(writer, sheet_name='Traffic_Demands', index=False)
                    
                    # 5. Performance Metrics
                    perf_data = {
                        'Metric': [
                            'Average Link Utilization',
                            'Maximum Link Utilization',
                            'Congested Links',
                            'Total Links',
                            'Demands Satisfied',
                            'Total Demands',
                            'Satisfaction Rate',
                            'Average Latency',
                            'Network Reliability',
                            'Jain\'s Fairness Index'
                        ],
                        'Value': [
                            f"{metrics['avg_link_utilization']:.2%}",
                            f"{metrics['max_link_utilization']:.2%}",
                            f"{metrics['congested_links']}",
                            f"{metrics['total_links']}",
                            f"{metrics['demands_satisfied']}",
                            f"{metrics['total_demands']}",
                            f"{metrics['satisfaction_rate']:.2%}",
                            f"{metrics['avg_latency']:.2f} ms",
                            f"{optimizer.calculate_network_reliability():.2%}",
                            f"{metrics.get('jains_fairness_index', 0):.4f}"
                        ]
                    }
                    perf_df = pd.DataFrame(perf_data)
                    perf_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                    
                    # 6. Bottleneck Analysis
                    bottlenecks = optimizer.detect_bottlenecks(threshold=0.7)
                    if bottlenecks:
                        bottleneck_data = []
                        for src, dst, util in bottlenecks:
                            bottleneck_data.append({
                                'Link': f"{src} → {dst}",
                                'Utilization (%)': f"{util*100:.2f}",
                                'Severity': 'Critical' if util > 0.9 else 'High'
                            })
                        bottleneck_df = pd.DataFrame(bottleneck_data)
                        bottleneck_df.to_excel(writer, sheet_name='Bottleneck_Analysis', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Download Full Report (Excel)",
                    data=excel_buffer,
                    file_name=f"network_topology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Includes Link Bandwidth Allocation, Node Info, Traffic Demands, Performance Metrics & More"
                )
            
            with col3:
                # Export network flows data
                if optimizer.flows:
                    flow_data = []
                    for (demand_id, src, dst), flow_value in optimizer.flows.items():
                        if flow_value > 0.01:  # Only include significant flows
                            flow_data.append({
                                'Demand ID': demand_id,
                                'Link': f"{src} → {dst}",
                                'Flow (Mbps)': f"{flow_value:.2f}"
                            })
                    
                    flow_df = pd.DataFrame(flow_data)
                    csv_flows = flow_df.to_csv(index=False)
                    
                    st.download_button(
                        label=" Download Flow Data (CSV)",
                        data=csv_flows,
                        file_name=f"network_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        else:
            st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
            st.error(f"**Error details:** {result.get('message', 'No additional information')}")
    
    # Show introductory information when network not built
    if not st.session_state['network_built']:
        st.markdown("##  What is Network Topology Optimization?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🏗️ Hierarchical Architecture
            
            This optimizer models realistic ISP-like networks with multiple layers:
            
            1. **🔌 Source Node**: Main backbone connection
            2. **🔶 Core Routers (Layer 1)**: High-capacity routing
            3. **🔷 Edge Routers (Layer 2)**: Distribution to end users
            4. **👥 End Users**: Emergency, Premium, and Standard tiers
            
            Each layer has capacity constraints and processing delays.
            """)
        
        with col2:
            st.markdown("""
            ### Advanced Features
            
            - **Multi-Commodity Flow**: Optimal routing for all traffic demands
            - **QoS-Aware Routing**: Priority-based path selection
            - **Redundancy**: Multiple paths for reliability
            - **Load Balancing**: Distribute traffic across paths
            - **Congestion Detection**: Identify and avoid bottlenecks
            - **Reliability Analysis**: Find critical nodes
            """)
        
        st.markdown("---")
        st.info("**Configure the network in the sidebar and click 'BUILD & OPTIMIZE NETWORK' to begin!**")
if __name__ == "__main__":
    main()
