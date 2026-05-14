#Genetic Algorithm . or Brute force bhi

import random
from typing import List,Tuple,Dict
from src.grid_model import Cell,Drone

#Drone ki specs
LIGHT_COST=1000
LIGHT_PAY=2.0
LIGHT_RNG=12
HEAVY_COST=1800
HEAVY_PAY=5.0
HEAVY_RNG=20

def calcFitness(nL:int,nH:int,totalDemand:float,budget:int)->Tuple[float,float,float]:
    
    cost=nL*LIGHT_COST+nH*HEAVY_COST
    if cost>budget or (nL+nH)==0:
        return -1.0,0.0,100.0  
  
    fleetCapacity=nL*LIGHT_PAY+nH*HEAVY_PAY  
    targetCapacity=20.0  
    covPct=min(fleetCapacity/targetCapacity*100,100.0)
    budPct=cost/budget*100
    score=0.75*(covPct/100)-0.25*(budPct/100)

    return score,covPct,budPct

def bruteForceSel(totalDemand:float,budget:int=10000)->Dict:
    #har combination try ho
    maxL=budget//LIGHT_COST
    maxH=budget//HEAVY_COST
    bestScore=-1;bestCombo=(0,0);bestCov=0;bestBud=0
    for nL in range(maxL+1):
        for nH in range(maxH+1):
            sc,cov,bud=calcFitness(nL,nH,totalDemand,budget)
            if sc>bestScore:
                bestScore=sc;bestCombo=(nL,nH);bestCov=cov;bestBud=bud
    return {"light":bestCombo[0],"heavy":bestCombo[1],
            "score":round(bestScore,4),"covPct":round(bestCov,1),"budPct":round(bestBud,1)}

def gaSelect(totalDemand:float,budget:int=10000,popSz:int=30,gens:int=50)->Dict:
    
    maxL=budget//LIGHT_COST
    maxH=budget//HEAVY_COST

    #initial population
    pop=[(random.randint(0,maxL),random.randint(0,maxH)) for _ in range(popSz)]

    def fitFn(chromo):
        sc,_,_=calcFitness(chromo[0],chromo[1],totalDemand,budget)
        return sc

    for g in range(gens):
       
        scored=[(c,fitFn(c)) for c in pop]
        scored.sort(key=lambda x:x[1],reverse=True)

        #top waley are kept
        eliteSz=max(2,popSz//5)
        newPop=[c for c,_ in scored[:eliteSz]]

        
        while len(newPop)<popSz:
            p1=random.choice(scored[:popSz//2])[0]
            p2=random.choice(scored[:popSz//2])[0]
            
            
            #single point crossover yahan par
            child=(p1[0],p2[1])
            #mutation mein 10 % chance
            if random.random()<0.1:
                child=(max(0,child[0]+random.randint(-1,1)),
                       max(0,child[1]+random.randint(-1,1)))
            newPop.append(child)
        pop=newPop

    #best extract hogaya
    best=max(pop,key=fitFn)
    sc,cov,bud=calcFitness(best[0],best[1],totalDemand,budget)
    return {"light":best[0],"heavy":best[1],
            "score":round(sc,4),"covPct":round(cov,1),"budPct":round(bud,1)}

def selectFleet(grid:List[List[Cell]],budget:int=10000,method:str="ga")->Tuple[Dict,List[Drone]]:
    
    totalDemand=sum(c.demand for row in grid for c in row)
    if method=="ga":
        res=gaSelect(totalDemand,budget)
    else:
        res=bruteForceSel(totalDemand,budget)

    #hub s
    hubs=[(c.row,c.col) for row in grid for c in row if c.isHub]

   
    drones=[]
    idx=1
    for i in range(res["light"]):
        hub=hubs[i%len(hubs)]
        drones.append(Drone(dId=f"D{idx}",dType="Light",cost=LIGHT_COST,
                            payload=LIGHT_PAY,maxRange=LIGHT_RNG,pos=hub,homeHub=hub))
        idx+=1
    for i in range(res["heavy"]):
        hub=hubs[(res["light"]+i)%len(hubs)]
        drones.append(Drone(dId=f"D{idx}",dType="Heavy",cost=HEAVY_COST,
                            payload=HEAVY_PAY,maxRange=HEAVY_RNG,pos=hub,homeHub=hub))
        idx+=1

    
   
    print("FLEET SELECTION")
    
    print(f"Method: {method.upper()}")
    print(f"Budget: ${budget}")
    print(f"Total demand: {totalDemand:.1f} kg")
    print(f"Selected: {res['light']} Light + {res['heavy']} Heavy drones")
    print(f"Total cost: ${res['light']*LIGHT_COST+res['heavy']*HEAVY_COST}")
    print(f"Coverage: {res['covPct']}% | Budget used: {res['budPct']}%")
    print(f"Fitness score: {res['score']}")
    print(f"Drones: {[d.dId+' ('+d.dType+')' for d in drones]}")
    
    return res,drones
