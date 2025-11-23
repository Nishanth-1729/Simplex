# Internet Bandwidth Allocation Optimization System
## Team Simplex - Advanced Convex Optimization Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)

## Project Overview

An interactive internet bandwidth allocation optimization system that uses convex optimization to efficiently and fairly distribute network bandwidth among multiple users. The system features a unified optimizer with multi-objective optimization, robust optimization capabilities, and network topology analysis.

###  Key Features

- **Data Generation**: Generate realistic user datasets with customizable parameters
- **Unified Optimizer**: All-in-one optimization combining fairness, efficiency, and robustness
  - Multi-objective optimization (fairness + efficiency + latency)
  - Robust optimization (box, budget, ellipsoidal uncertainty sets)
  - Convergence tracking and visualization
  - Multiple utility functions (log, sqrt, linear, alpha-fair)
- **Network Topology Optimizer**: Hierarchical network optimization with multi-commodity flow
  - 4-layer network architecture (Core → Distribution → Aggregation → Access)
  - QoS-aware routing with bandwidth guarantees
  - Bottleneck detection and network visualization
- **Interactive Dashboard**: Streamlit-based web interface with real-time visualizations
- **High Performance**: Fast optimization with CVXPY solvers (ECOS, SCS, CVXOPT)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Mathematical Formulation](#mathematical-formulation)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Application Modules](#application-modules)
- [Performance Metrics](#performance-metrics)
- [Team](#team)

## Problem Statement

In modern networks (homes, offices, ISPs, data centers), internet bandwidth is a **scarce shared resource** that must be allocated among multiple competing users and applications. Poor allocation leads to:

- **Unfairness**: Some users get too much, others too little
- **Inefficiency**: Total network utility is suboptimal
- **Congestion**: Network bottlenecks and high latency
- **Poor QoS**: Critical applications starved of bandwidth

This project solves the bandwidth allocation problem using **convex optimization**, ensuring:

- **Optimal Efficiency**: Maximize total network utility
- **Fairness**: Equitable distribution (Jain's index > 0.9)
- **Priority Handling**: VIP users and critical applications get preference
- **Constraint Satisfaction**: Minimum/maximum bandwidth guarantees
- **Real-Time Adaptation**: Dynamic reallocation as demands change

## Mathematical Formulation

### Basic Formulation

```
Maximize:   Σᵢ wᵢ · Uᵢ(xᵢ)

Subject to: Σᵢ xᵢ ≤ C                    (Capacity constraint)
            xᵢ,min ≤ xᵢ ≤ xᵢ,max         (Bandwidth bounds)
            xᵢ ≥ 0                        (Non-negativity)
```

**Where:**
- `xᵢ` = Bandwidth allocated to user i (decision variable)
- `wᵢ` = Priority weight for user i
- `Uᵢ(xᵢ)` = Utility function (log, sqrt, or linear)
- `C` = Total available bandwidth capacity
- `xᵢ,min`, `xᵢ,max` = Min/max bandwidth constraints

### Utility Functions

1. **Logarithmic (Proportional Fairness)**: `U(x) = log(x)`
2. **Square Root (Balanced Fairness)**: `U(x) = √x`
3. **Linear (Maximum Efficiency)**: `U(x) = x`
4. **Alpha-Fair**: `U(x) = x^(1-α)/(1-α)` for α ∈ [0, ∞)

### Multi-Objective Formulation

```
Maximize:   [U_fairness, U_efficiency, U_latency]

Subject to: Network constraints
            QoS constraints
            Fairness constraints
```

## System Architecture

```
Simplex_code/
├──  frontend.py                 # Main Streamlit application 
│
├──  backend/
│   ├── unified_optimizer.py       # All-in-one optimizer (multi-objective + robust)
│   ├── convergence_visualizer.py # Convergence tracking and plotting
│   ├── visualizer.py              # Bandwidth allocation visualizations
│   ├── network_topology_optimizer.py  # Hierarchical network optimization
│   ├── network_visualizer.py      # Network topology visualizations
│   ├── data_generator.py          # User data generation utilities
│   └── __init__.py
│
├── 📄 requirements.txt            # Python dependencies
├── 📖 README.md                   # This file
```
### Dependencies

```
# Core optimization
cvxpy>=1.4.0
numpy>=1.24.0

# Visualization
plotly>=5.17.0
networkx>=3.2

# Web Dashboard
streamlit>=1.28.0

# Utilities
pandas>=2.0.0
```

See `requirements.txt` for complete dependency list.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Nishanth-1729/Simplex.git
cd Simplex
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install requirements
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
# Start the Streamlit app
streamlit run frontend.py
```

Then open `http://localhost:8501` in your browser.

### 4. Using the Application

1. **Data Generation Page**: Generate or upload user datasets with customizable parameters
2. **Unified Optimizer Page**: Run bandwidth allocation optimization with various settings
3. **Network Topology Page**: Analyze hierarchical network structure and flows
4. **Guide Page**: View application documentation and usage instructions

##  Application Modules

### 1. Data Generation

Generate realistic user datasets for optimization testing:

- **Manual Dataset Creation**: Define custom user parameters
- **Random Generation**: Generate datasets with configurable size and characteristics
- **CSV Upload/Download**: Import existing data or export generated datasets
- **Parameter Controls**: Adjust demands, priorities, min/max bandwidths

### 2. Unified Optimizer

All-in-one bandwidth allocation optimizer with multiple optimization modes:

**Standard Optimization:**
- Utility functions: Logarithmic (proportional fairness), square root, linear, alpha-fair
- Customizable total capacity and user priorities
- Real-time convergence tracking

**Multi-Objective Optimization:**
- Simultaneous optimization of fairness, efficiency, and latency
- Adjustable objective weights
- Pareto frontier exploration

**Robust Optimization:**
- Handle demand uncertainty with three uncertainty set models:
  - Box uncertainty (±Δ bounds)
  - Budget uncertainty (limited total deviation)
  - Ellipsoidal uncertainty (probabilistic bounds)
- Worst-case performance guarantees

**Visualizations:**
- Bandwidth allocation bar charts
- Convergence plots (objective value, fairness, constraint violations)
- Fairness metrics and utilization statistics

### 3. Network Topology Optimizer

Hierarchical network structure optimization:

- **4-Layer Architecture**: Core → Distribution → Aggregation → Access
- **Multi-Commodity Flow**: Optimal routing with capacity constraints
- **QoS-Aware Routing**: Bandwidth guarantees and latency requirements
- **Bottleneck Detection**: Identify congested network links
- **Network Visualization**: 3D interactive network graphs with flow paths
- **Performance Analysis**: Link utilization, flow distribution, latency metrics

### 4. User Guide

In-app documentation with:
- Feature overview for each module
- Quick start instructions
- FAQ and troubleshooting tips
- Technical information about algorithms

## Technical Details

### Optimization Solvers

The system uses CVXPY with multiple backend solvers:
- **ECOS**: Default solver for convex problems
- **SCS**: Splitting conic solver for large-scale problems
- **CVXOPT**: Alternative solver for general convex optimization

### Fairness Metrics

- **Jain's Fairness Index**: Measures allocation equity (0 to 1, higher is better)
- **Max-Min Ratio**: Ratio of largest to smallest allocation
- **Coefficient of Variation**: Standard deviation relative to mean

### Utility Functions

1. **Logarithmic**: `U(x) = log(x)` - Proportional fairness
2. **Square Root**: `U(x) = √x` - Balanced fairness
3. **Linear**: `U(x) = x` - Maximum efficiency
4. **Alpha-Fair**: `U(x) = x^(1-α)/(1-α)` - Tunable fairness parameter

## Performance Metrics

The application provides real-time metrics:

### Optimization Metrics
- **Objective Value**: Total system utility
- **Fairness Index**: Jain's fairness index (0 to 1)
- **Utilization**: Percentage of total capacity used
- **Solve Time**: Optimization computation time
- **Solver Status**: Optimal, infeasible, or unbounded

### Network Metrics (Topology Module)
- **Total Flow**: Sum of all commodities routed
- **Link Utilization**: Usage percentage per network link
- **Bottleneck Detection**: Identification of congested links
- **Latency Analysis**: End-to-end delay calculations

## Visualizations

The application includes interactive visualizations:

1. **Bandwidth Allocation Bar Charts**: User-wise bandwidth distribution with priorities
2. **Convergence Plots**: Iteration-by-iteration tracking of objective value, fairness, and constraints
3. **3D Network Graphs**: Interactive hierarchical network topology with flow paths
4. **Flow Heatmaps**: Link utilization and traffic patterns
5. **Performance Gauges**: Real-time metrics display

## 📄 License

MIT License - see LICENSE file for details
