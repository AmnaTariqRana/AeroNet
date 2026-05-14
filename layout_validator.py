

#CSP; Arc Consistency
from collections import deque
from typing import List,Dict,Tuple,Set
from src.grid_model import Cell,getNeighbors,manhattan



def constraintR1(zi:str,zj:str)->bool:
    
    if zi=="Industrial" and zj in {"School","Hospital"}:
        return False
    if zj=="Industrial" and zi in {"School","Hospital"}:
        return False
    return True

def constraintR2(ci:Tuple,hubs:List)->bool:
    
    minDist=min((manhattan(ci,h) for h in hubs),default=999)
    return minDist<=3

def constraintR3(ci:Tuple,chPads:List)->bool:
    
    minDist=min((manhattan(ci,cp) for cp in chPads),default=999)
    return minDist<=2

def constraintR4(hosps:List,medPts:List)->bool:
    
    for h in hosps:
        for mp in medPts:
            if manhattan(h,mp)<=1:
                return True
    return False


def revise(domains:Dict,xi:Tuple,xj:Tuple)->bool:
   
    revised=False
    toRemove=set()
    for vi in list(domains[xi]):
        
        if not any(constraintR1(vi,vj) for vj in domains[xj]):
            toRemove.add(vi)
            revised=True
    domains[xi]-=toRemove
    return revised

def ac3(grid:List[List[Cell]])->Tuple[Dict,List[str]]:
    
    #domains
    domains:Dict[Tuple,Set[str]]={}
    for row in grid:
        for c in row:
            domains[(c.row,c.col)]={c.zone}

    
    hubs=[(c.row,c.col) for row in grid for c in row if c.isHub]
    chPads=[(c.row,c.col) for row in grid for c in row if c.isCharging]
    medPts=[(c.row,c.col) for row in grid for c in row if c.isMedPickup]
    hosps=[(c.row,c.col) for row in grid for c in row if c.zone=="Hospital"]

    #arc queue
    queue=deque()
    seen=set()
    for row in grid:
        for cell in row:
            xi=(cell.row,cell.col)
            for nr,nc in getNeighbors(cell.row,cell.col):
                xj=(nr,nc)
                if (xi,xj) not in seen:
                    seen.add((xi,xj))
                    seen.add((xj,xi))
                    queue.append((xi,xj))
                    queue.append((xj,xi))

    arcChecks=0
    errs=[]

    
    while queue:
        xi,xj=queue.popleft()
        arcChecks+=1
        if revise(domains,xi,xj):
            if not domains[xi]:
                errs.append(f"R1 FAIL: domain of ({xi[0]},{xi[1]}) emptied -- "
                            f"Industrial adjacent to School/Hospital at ({xj[0]},{xj[1]})")
            else:
                #rechecking all neighbours of Xi but not Xj
                for nr,nc in getNeighbors(xi[0],xi[1]):
                    xk=(nr,nc)
                    if xk!=xj:
                        queue.append((xk,xi))

    
    for row in grid:
        for cell in row:
            ci=(cell.row,cell.col)
            #R2
            if cell.zone=="Residential" and not constraintR2(ci,hubs):
                minD=min(manhattan(ci,h) for h in hubs)
                errs.append(f"R2 FAIL: Residential({cell.row},{cell.col}) nearest hub={minD} cells (limit=3)")
            #R3
            if cell.isHub and not constraintR3(ci,chPads):
                minD=min(manhattan(ci,cp) for cp in chPads)
                errs.append(f"R3 FAIL: Hub({cell.row},{cell.col}) nearest pad={minD} cells (limit=2)")

    #R4
    if not constraintR4(hosps,medPts):
        errs.append("R4 FAIL: No Hospital has a Medical Pickup within 1 cell")

    return domains,errs,arcChecks


def validateLayout(grid:List[List[Cell]])->Dict:
   
    domains,errs,arcChecks=ac3(grid)
    ruleErrs={"R1":[],"R2":[],"R3":[],"R4":[]}
    for e in errs:
        for r in ["R1","R2","R3","R4"]:
            if r in e:
                ruleErrs[r].append(e)
                break

    passed=[r for r in ["R1","R2","R3","R4"] if not ruleErrs[r]]
    failed=[r for r in ["R1","R2","R3","R4"] if ruleErrs[r]]
    valid=len(failed)==0

    results={"passed":passed,"failed":failed,"errors":errs,"domains":domains}

    
    print("LAYOUT VALIDATION")
   
    print(f"Algorithm     : AC-3 (Arc Consistency 3)")
    print(f"Variables     : {len(domains)} cells")
    print(f"Arcs processed: {arcChecks}")
    print(f"Layout valid  : {valid}")
    print(f"Passed rules  : {', '.join(passed) if passed else 'None'}")
    print(f"Failed rules  : {', '.join(failed) if failed else 'None'}")
    if errs:
        print("\nViolations:")
        for e in errs:
            print(f"  - {e}")
    

    return results