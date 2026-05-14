#main file. sari files run here

import sys
import os
import matplotlib
matplotlib.use("Agg") 


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_model import buildGrid,printGrid
from src.layout_validator import validateLayout
from src.fleet_selector import selectFleet
from src.astar_planner import astar,planDeliveryRoute
from src.delivery_simulator import SimEngine
from src.ml_pipeline import trainDemandModel,trainAnomalyClassifier,runFullPipeline
from src.visualization import (plotZoneMap,plotRouteMap,plotDemandHeatmap,
                                plotAnomalyTable,plotEventLog,saveAllPlots)

def main():
    
    print("AERONET LITE: Autonomous Drone Delivery Simulation")

    #grid
    print("\n[1] Building 10x10 city grid...")
    grid=buildGrid()
    printGrid(grid)

    #CSP
    print("\n[2] Validating layout (CSP)...")
    valRes=validateLayout(grid)

    #Genetic Algorithm
    print("\n[3] Selecting fleet (Genetic Algorithm)...")
    fleetInfo,drones=selectFleet(grid,budget=10000,method="ga")

    #A* 
    print("\n[4] Testing A* path planner...")
    hubs=[(c.row,c.col) for row in grid for c in row if c.isHub]
    testStart=hubs[0]
    testGoal=(6,6)
    path,cost,msg=astar(testStart,testGoal,grid)
    print(f"  Test route: {testStart} -> {testGoal}")
    print(f"  {msg}")
    if path:
        print(f"  Path: {path}")

    # Full delivery route
    print("\n[5] Testing full delivery route (hub->pickup->dropoff->hub)...")
    fullPath,fullCost,msgs=planDeliveryRoute(hubs[0],(3,5),(7,2),grid)
    for m in msgs:
        print(f"  {m}")

    #ML classifiers
    print("\n[6] Running ML Pipeline...")
    demandRes,anomalyRes=runFullPipeline()

    #20 Step Simulation
    print("\n[7] Running 20-step simulation...")
    sim=SimEngine(grid,drones,fleetInfo)

    
    def forecaster():
        bestMdl=min(demandRes.items(),key=lambda x:x[1]["rmse"])
        return bestMdl[1]["rmse"]

    sim.run20Steps(forecaster=forecaster)

    #visuals
    print("\n[8] Generating visualizations...")
    figDir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"report","figures")
    os.makedirs(figDir,exist_ok=True)

    #routes taken
    routes=[d.route for d in sim.deliveries if d.route]
    dLabels=[d.assignedDrone for d in sim.deliveries if d.route]

    plotZoneMap(grid,savePath=os.path.join(figDir,"zone_map.png"),show=False)
    plotRouteMap(grid,routes,dLabels,savePath=os.path.join(figDir,"route_map.png"),show=False)
    plotDemandHeatmap(grid,savePath=os.path.join(figDir,"demand_heatmap.png"),show=False)
    plotAnomalyTable(sim.anomalies,savePath=os.path.join(figDir,"anomaly_log.png"),show=False)
    plotEventLog(sim.log,savePath=os.path.join(figDir,"event_log.png"),show=False)

    print(f"\n[DONE] All figures saved to {figDir}/")
    
    print("AERONET LITE EXECUTION COMPLETE")
    

if __name__=="__main__":
    main()
