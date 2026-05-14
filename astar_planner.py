import heapq
from typing import List,Tuple,Optional,Dict
from src.grid_model import Cell,getNeighbors,manhattan

def astar(start:Tuple[int,int],goal:Tuple[int,int],
          grid:List[List[Cell]])->Tuple[Optional[List[Tuple[int,int]]],float,str]:

    rows=len(grid);cols=len(grid[0])

    if grid[start[0]][start[1]].noFly:
        return None,0.0,f"Start {start} is a no-fly cell"
    if grid[goal[0]][goal[1]].noFly:
        return None,0.0,f"Goal {goal} is a no-fly cell"

    openSet=[]
    heapq.heappush(openSet,(0,0,start))
    cameFrom:Dict[Tuple[int,int],Tuple[int,int]]={}
    gScore:Dict[Tuple[int,int],float]={start:0}
    visited=set()

    while openSet:
        f,g,curr=heapq.heappop(openSet)

        if curr==goal:
            path=[]
            node=curr
            while node in cameFrom:
                path.append(node)
                node=cameFrom[node]
            path.append(start)
            path.reverse()
            return path,gScore[goal],f"Path found: {len(path)} cells, cost={gScore[goal]:.1f}"

        if curr in visited:
            continue
        visited.add(curr)

        for nr,nc in getNeighbors(curr[0],curr[1],rows):
            nbr=(nr,nc)
            if grid[nr][nc].noFly:
                continue  
            moveCost=0.8 if grid[nr][nc].zone=="Commercial" else 1.0
            tentG=gScore[curr]+moveCost
            if tentG<gScore.get(nbr,float('inf')):
                gScore[nbr]=tentG
                cameFrom[nbr]=curr
                h=manhattan(nbr,goal)
                heapq.heappush(openSet,(tentG+h,tentG,nbr))

    return None,0.0,f"No safe path from {start} to {goal}"

def planDeliveryRoute(hub:Tuple[int,int],pickup:Tuple[int,int],
                      dropoff:Tuple[int,int],grid:List[List[Cell]]
                      )->Tuple[Optional[List[Tuple[int,int]]],float,List[str]]:
    
    msgs=[]
    fullPath=[]
    totalCost=0.0

    p1,c1,m1=astar(hub,pickup,grid)
    msgs.append(f"Hub->Pickup: {m1}")
    if p1 is None:
        return None,0.0,msgs

    p2,c2,m2=astar(pickup,dropoff,grid)
    msgs.append(f"Pickup->Dropoff: {m2}")
    if p2 is None:
        return None,0.0,msgs

    p3,c3,m3=astar(dropoff,hub,grid)
    msgs.append(f"Dropoff->Hub: {m3}")
    if p3 is None:
        return None,0.0,msgs

    fullPath=p1+p2[1:]+p3[1:]
    totalCost=c1+c2+c3
    msgs.append(f"Total route: {len(fullPath)} cells, cost={totalCost:.1f}")
    return fullPath,totalCost,msgs