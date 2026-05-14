# AeroNet Lite

**Autonomous Drone Delivery Simulation**

---

## Overview

AeroNet Lite simulates a drone delivery system on a 10x10 city grid. It integrates five AI modules:

| Module | Technique |
|--------|-----------|
| Layout Validator | Constraint Satisfaction Problem (CSP) |
| Fleet Selector | Genetic Algorithm (GA) |
| Path Planner | A* Search |
| Disruption Handler | Real-time A* Rerouting |
| ML Pipeline | Regression + Classification |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Run

```bash
cd aeronet_lite
python -m src.main
```

## Project Structure

```
aeronet_lite/
  data/raw/              # raw datasets
  data/processed/        # cleaned data
  src/
    grid_model.py        # shared 10x10 grid model
    layout_validator.py  # CSP constraint checker
    fleet_selector.py    # GA fleet optimizer
    astar_planner.py     # A* path planner
    delivery_simulator.py# 20-step simulation engine
    ml_pipeline.py       # demand forecast + anomaly detection
    visualization.py     # matplotlib visualizations
    main.py              # entry point
  notebooks/
    demand_forecasting.ipynb
    anomaly_classifier.ipynb
  report/figures/        # saved plots
  README.md
```

## Modules Detail

### Module 1: Layout Validator (CSP)
Checks 4 rules: industrial safety (R1), residential coverage (R2), hub charging (R3), medical access (R4).

### Module 2: Fleet Selector (GA)
Selects Light (1000$/2kg/12cells) and Heavy (1800$/5kg/20cells) drones under budget.
Fitness: `0.75 * coverage% - 0.25 * budgetUsed%`

### Module 3: A* Path Planner
Routes: hub -> pickup -> dropoff -> hub. Manhattan heuristic. Commercial corridors cost 0.8.

### Module 4: Disruption Handler
Activates no-fly cells mid-simulation. Reroutes affected drones via A*.

### Module 5: ML Pipeline
- **Regression**: Linear Regression + Random Forest on synthetic Bike Sharing data. Reports MAE/RMSE.
- **Classification**: Decision Tree + Random Forest + KNN on synthetic anomaly data. Reports accuracy + confusion matrix.

## Simulation (20 Steps)
Steps 1-3: Init, validate, fleet. Steps 4-6: Generate deliveries. Steps 7-10: Move drones.
Step 11: Disrupt. Steps 12-14: Reroute. Steps 15-17: Forecast. Step 18: Anomaly. Step 20: Summary.

## Datasets
- Bike Sharing Demand Data
- Synthetic anomaly data 
- Population density Data from kaggle
- Real datasets can be placed in `data/raw/` if available

## Output
- Console event log for all 20 steps
- Saved figures: zone_map, route_map, demand_heatmap, anomaly_log, event_log
