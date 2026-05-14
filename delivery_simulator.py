import random
from typing import List, Dict, Tuple
from src.grid_model import Cell, Drone, Delivery, manhattan
from src.astar_planner import astar, planDeliveryRoute

class SimEngine:

    def __init__(self, grid: List[List[Cell]], drones: List[Drone], fleetInfo: Dict):
        self.grid=grid
        self.drones=drones
        self.fleetInfo=fleetInfo
        self.deliveries: List[Delivery]=[]
        self.log: List[str]=[]
        self.step=0
        self.completed=0
        self.delayed=0
        self.failed=0
        self.anomalies: List[Dict] = []

    def addLog(self, msg: str):
        entry=f"Step {self.step}: {msg}"
        self.log.append(entry)
        print(entry)

    def genDeliveries(self, n: int = 8) -> List[Delivery]:
        validCells=[(c.row, c.col) for row in self.grid for c in row if not c.noFly]
        hubs=[(c.row, c.col) for row in self.grid for c in row if c.isHub]
        dels=[]
        startIdx=len(self.deliveries) + 1
        for i in range(n):
            pk=random.choice(validCells)
            dp=random.choice([v for v in validCells if v != pk])
            wt=round(random.uniform(0.5, 4.5), 1)
            d=Delivery(delId=f"DEL{startIdx+i}", pickup=pk, dropoff=dp, weight=wt)
            dels.append(d)
        self.deliveries.extend(dels)
        return dels

    def assignDrones(self):
        for dl in self.deliveries:
            if dl.status in ("completed", "inprogress", "failed"):
                continue
            bestDrn=None
            bestDist=999
            for drn in self.drones:
                if drn.status!="idle":
                    continue
                if drn.payload<dl.weight:
                    continue
                dist = manhattan(drn.pos, dl.pickup)
                if dist < bestDist:
                    bestDist = dist
                    bestDrn = drn
            if bestDrn:
                dl.assignedDrone = bestDrn.dId
                bestDrn.status="enroute"
                route,cost,msgs=planDeliveryRoute(bestDrn.homeHub, dl.pickup, dl.dropoff, self.grid)
                if route:
                    dl.route=route
                    dl.routeCost=cost
                    dl.status="inprogress"
                    self.addLog(f"{dl.delId} assigned to {bestDrn.dId} ({bestDrn.dType}). Route cost={cost:.1f}")
                else:
                    dl.status="failed"
                    bestDrn.status="idle"
                    self.failed+=1
                    self.addLog(f"{dl.delId} FAILED: no safe route. {msgs[-1]}")
            else:
                dl.status="delayed"
                self.delayed+=1
                self.addLog(f"{dl.delId} DELAYED: no available drone (weight={dl.weight}kg)")

    def moveDrones(self, steps: int = 3):
        for dl in self.deliveries:
            if dl.status!="inprogress":
                continue
            drn=next((d for d in self.drones if d.dId == dl.assignedDrone), None)
            if not drn:
                continue
            for _ in range(steps):
                if len(dl.route) <= 1:
                    dl.status="completed"
                    drn.status="idle"
                    drn.pos = drn.homeHub
                    self.completed += 1
                    self.addLog(f"{dl.delId} completed by {drn.dId}")
                    break
                dl.route.pop(0)
                drn.pos=dl.route[0]
                drn.battery-=random.uniform(1.5, 3.0)

    def activateNoFly(self, r: int, c: int):
        self.grid[r][c].noFly = True
        self.addLog(f"No-fly cell activated at ({r},{c})")
        # check affected deliveries
        for dl in self.deliveries:
            if dl.status!="inprogress":
                continue
            if (r, c) in dl.route:
                drn=next((d for d in self.drones if d.dId == dl.assignedDrone), None)
                if not drn:
                    continue
                remaining=dl.route[dl.route.index(drn.pos):]
                if len(remaining)<2:
                    continue
                target=remaining[-1]
                newPath, newCost, msg=astar(drn.pos, target, self.grid)
                if newPath:
                    dl.route=newPath
                    dl.routeCost=newCost
                    self.addLog(f"{drn.dId} rerouted via A*. New cost={newCost:.1f}")
                else:
                    dl.status="failed"
                    drn.status="idle"
                    self.failed+=1
                    self.addLog(f"{drn.dId} cannot reach destination safely. {dl.delId} FAILED")

    def injectAnomaly(self, droneId: str, anomType: str = "battery"):
        
        drn=next((d for d in self.drones if d.dId == droneId), None)
        if not drn:
            return
        anom={"drone": droneId, "type": anomType, "step": self.step}
        if anomType=="battery":
            drn.battery-=40 
            anom["detail"]=f"Battery dropped to {drn.battery:.1f}%"
        elif anomType=="route":
            anom["detail"]="Drone deviated from planned path"
        elif anomType=="sensor":
            anom["detail"]="Altitude sensor spike detected"
        self.anomalies.append(anom)
        self.addLog(f"{anomType.upper()} anomaly detected for {droneId}. {anom['detail']}")

        if drn.battery<20:
            drn.status="grounded"
            self.addLog(f"{droneId} grounded - returning to hub {drn.homeHub}")
            for dl in self.deliveries:
                if dl.assignedDrone == droneId and dl.status == "inprogress":
                    dl.status = "delayed"
                    self.delayed += 1
                    self.addLog(f"{dl.delId} DELAYED due to {droneId} grounding")

    def printSummary(self):
        self.completed = sum(1 for d in self.deliveries if d.status == "completed")
        self.delayed = sum(1 for d in self.deliveries if d.status == "delayed")
        self.failed = sum(1 for d in self.deliveries if d.status == "failed")
        inProg = sum(1 for d in self.deliveries if d.status == "inprogress")
       
        print("SIMULATION SUMMARY")
        
        print(f"Total deliveries: {len(self.deliveries)}")
        print(f"Completed: {self.completed}")
        print(f"In progress: {inProg}")
        print(f"Delayed: {self.delayed}")
        print(f"Failed: {self.failed}")
        print(f"Anomalies detected: {len(self.anomalies)}")
        print(f"Active no-fly cells: {sum(1 for row in self.grid for c in row if c.noFly)}")
        

    def run20Steps(self, forecaster=None, classifier=None):

        
        print("AERONET LITE - 20 STEP SIMULATION")
        

        self.step=1
        from src.layout_validator import validateLayout
        res=validateLayout(self.grid)
        self.addLog(f"Layout validation: {'PASSED' if not res['failed'] else 'FAILED ('+','.join(res['failed'])+')'}")

        self.step=2
        self.addLog(f"Grid initialized (10x10). Zones loaded.")

        self.step=3
        self.addLog(f"Fleet selected: {self.fleetInfo['light']} light, {self.fleetInfo['heavy']} heavy drones")

        self.step=4
        dels=self.genDeliveries(8)
        self.addLog(f"Generated {len(dels)} deliveries")

        self.step=5
        self.assignDrones()

        self.step=6
        self.addLog(f"All routes computed. {sum(1 for d in self.deliveries if d.status=='inprogress')} active deliveries")


        for s in range(7, 11):
            self.step=s
            self.moveDrones(steps=4)
            activeD=sum(1 for d in self.deliveries if d.status == "inprogress")
            self.addLog(f"Drones advanced. {activeD} active, {self.completed} completed")

        self.step=11
        valid_cells=[(r, c) for r in range(10) for c in range(10) 
                       if not self.grid[r][c].noFly and not self.grid[r][c].isHub]
        if valid_cells:
            rand_r, rand_c = random.choice(valid_cells)
            self.activateNoFly(rand_r, rand_c)
        else:
            self.addLog("No valid cells available for no-fly activation")

        for s in range(12, 15):
            self.step = s
            self.moveDrones(steps=4)
            activeD = sum(1 for d in self.deliveries if d.status == "inprogress")
            self.addLog(f"Post-disruption movement. {activeD} active")

        self.step=15
        if forecaster:
            pred=forecaster()
            self.addLog(f"Demand forecast: predicted avg demand={pred:.1f}")
        else:
            self.addLog("Demand forecast: ML model predicts moderate demand")

        self.step=16
        extraDel=self.genDeliveries(1)
        self.deliveries.extend(extraDel)
        self.assignDrones()
        self.addLog(f"Forecast-driven delivery added: {extraDel[0].delId}")

        self.step=17
        self.moveDrones(steps=4)
        self.addLog("Additional deliveries progressed")

        self.step=18
        activeDrones = [d for d in self.drones if d.status == "enroute"]
        if activeDrones:
            self.injectAnomaly(activeDrones[0].dId, "battery")
        elif self.drones:
            self.injectAnomaly(self.drones[0].dId, "sensor")
        else:
            self.addLog("No drones available for anomaly injection")

        self.step=19
        self.moveDrones(steps=4)
        self.addLog("Post-anomaly adjustments complete")

        self.step=20
        self.addLog("Simulation complete")
        self.printSummary()

        return self.log