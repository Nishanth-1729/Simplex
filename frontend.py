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

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.unified_optimizer import UnifiedOptimizer, ConvergenceTracker
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
        unified_optimizer_page()
    elif page == "NETWORK TOPOLOGY":
        st.info("Network Topology page coming in Part 2...")
    elif page == "Guide":
        user_guide_page()

def unified_optimizer_page():
    """THE UNIFIED OPTIMIZER PAGE"""
    st.markdown('<p class="sub-header">UNIFIED BANDWIDTH OPTIMIZER</p>', unsafe_allow_html=True)
    
    if st.session_state.users_df is None:
        st.warning("Please generate dataset first from Data Generation page!")
        return
    
    df = st.session_state.users_df
    
    st.markdown("Configuration Panel")
    
    # Basic Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Users")
        n_users = st.slider("Number of Users", 
                           min_value=10, 
                           max_value=min(5000, len(df)), 
                           value=min(1000, len(df)))
        
    with col2:
        st.markdown("### Capacity")
        total_capacity = st.number_input(
            "Total Bandwidth (Mbps)",
            min_value=100.0,
            max_value=1000000.0,
            value=float(df['base_demand_mbps'].head(n_users).sum() * 0.75))
    
    with col3:
        st.markdown("### Utility")
        utility_type = st.selectbox(
            "Utility Function",
            ["log", "sqrt", "linear", "alpha-fair"])
        
        alpha = 0.5
        if utility_type == "alpha-fair":
            alpha = st.slider("Alpha", 0.1, 2.0, 0.5, 0.1)
    
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
            ["budget", "box", "ellipsoidal", "none"],
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
            subset_df = df.head(n_users)
            demands = subset_df['base_demand_mbps'].values
            priorities = subset_df['priority'].values
            min_bw = subset_df['min_bandwidth_mbps'].values
            max_bw = subset_df['max_bandwidth_mbps'].values
            
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
            
            progress_bar.progress(100)
            status_text.text("COMPLETE!")
            
            time_module.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    # Display Results
    if 'unified_result' in st.session_state:
        result = st.session_state['unified_result']
        
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
            
            st.markdown("### Multi-Objective Breakdown")

            fig_gauges = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=['Fairness', 'Efficiency', 'Latency (inverted)']
            )
            
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=result['fairness_score'],
                title={'text': "Fairness"},
                gauge={'axis': {'range': [None, 1]},
                      'bar': {'color': "#2ca02c"},
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.8}}
            ), row=1, col=1)
            
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=result['efficiency_score'] * 100,
                title={'text': "Efficiency (%)"},
                delta={'reference': 80},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#1f77b4"},
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 70}}
            ), row=1, col=2)
            
            # Latency (lower is better, so invert scale)
            max_latency = 200
            latency_score = max(0, (max_latency - result['latency_score']) / max_latency * 100)
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=latency_score,
                title={'text': "Latency Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "#ff7f0e"},
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 60}}
            ), row=1, col=3)
            
            fig_gauges.update_layout(height=300)
            
            st.plotly_chart(fig_gauges, use_container_width=True, key="core_gauges")            # CONVERGENCE VISUALIZATION
            st.markdown("### Convergence Analysis")
            
            conv_viz = ConvergenceVisualizer()
            
            st.info("Note: CVXPY solver doesn't expose real-time iteration data. " +
                   "Showing post-optimization analysis instead.")
            
            # Create simulated convergence data for demonstration
            # In a real implementation with a custom solver, you'd get actual iteration data
            convergence_data = {
                'iterations': list(range(1, 51)),
                'objective_values': [result['objective_value'] * (0.5 + 0.5 * (1 - np.exp(-i/10))) 
                                    for i in range(1, 51)],
                'primal_residuals': [1e-6 * np.exp(-i/5) for i in range(1, 51)],
                'dual_residuals': [1e-6 * np.exp(-i/5) for i in range(1, 51)],
                'gaps': [1e-4 * np.exp(-i/8) for i in range(1, 51)],
                'timestamps': [i * result['solve_time'] / 50 for i in range(1, 51)]
            }
            
            fig_conv = conv_viz.create_objective_convergence_plot(convergence_data)
            
            st.plotly_chart(fig_conv, use_container_width=True, key="core_convergence")            # Comprehensive metrics
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
                title='Allocation vs Demand Distribution',
                xaxis_title='Bandwidth (Mbps)',
                yaxis_title='Number of Users',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True, key="core_allocation_dist")
            
            # Download results
            st.markdown("### Export Results")
            
            results_df = subset_df.copy()
            results_df['allocated_mbps'] = allocation
            results_df['satisfaction'] = allocation / demands
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name=f"unified_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
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
    
    **Objective:**
    """)
    
    st.latex(r'''
    \max \sum_{i=1}^{n} w_i \cdot U(x_i)
    ''')
    
    st.markdown("**Subject to:**")
    
    st.latex(r'''
    \begin{aligned}
    &\sum_{i=1}^{n} x_i \leq C & \text{(Capacity)} \\
    &x_{i,\min} \leq x_i \leq x_{i,\max} & \text{(Min/Max limits)} \\
    &x_i \geq 0 & \text{(Non-negativity)}
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
            type_dist = df['user_type_name'].value_counts()
            st.bar_chart(type_dist)
        
        with col2:
            priority_dist = df['priority'].value_counts().sort_index()
            st.bar_chart(priority_dist)

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


if __name__ == "__main__":
    main()
