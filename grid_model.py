

#1=Low, 2=Medium, 3=High.


from dataclasses import dataclass,field
from typing import List,Dict,Tuple
import os,pandas as pd


@dataclass
class Cell:
    row:int=0
    col:int=0
    zone:str="OpenField"           
    density:int=0                  
    isHub:bool=False
    isCharging:bool=False
    isMedPickup:bool=False
    noFly:bool=False
    demand:float=0.0              


@dataclass
class Drone:
    dId:str=""
    dType:str="Light"
    cost:int=1000
    payload:float=2.0
    maxRange:int=12
    pos:Tuple[int,int]=(0,0)
    homeHub:Tuple[int,int]=(0,0)
    battery:float=100.0
    status:str="idle"


@dataclass
class Delivery:
    
    delId:str=""
    pickup:Tuple[int,int]=(0,0)
    dropoff:Tuple[int,int]=(0,0)
    weight:float=1.0
    assignedDrone:str=""
    status:str="pending"
    route:List[Tuple[int,int]]=field(default_factory=list)
    routeCost:float=0.0



def loadDensityLevels()->Tuple[int,int,pd.DataFrame]:
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csvPath=os.path.join(base,"data","raw","uscitypopdensity.csv")
    df=pd.read_csv(csvPath,encoding="utf-8-sig")
    col="Population Density (Persons/Square Mile)"
    
    
    lowCut=int(df[col].quantile(0.33))  
    highCut=int(df[col].quantile(0.66)) 
    return lowCut,highCut,df

def classifyDensity(rawVal:int,lowCut:int,highCut:int)->int:

    if rawVal<lowCut:
        return 1
    elif rawVal<=highCut:
        return 2
    else:
        return 3


def buildGrid()->List[List[Cell]]:
    
    #zone
    zMap=[
        ["Commercial","Commercial","Residential","Residential","OpenField","OpenField","Residential","Residential","Commercial","Commercial"],
        ["Commercial","Hospital","Residential","Residential","School","OpenField","Residential","Residential","Industrial","Commercial"],
        ["Residential","Residential","Residential","OpenField","OpenField","OpenField","Residential","Residential","Industrial","Industrial"],
        ["Residential","Residential","OpenField","Commercial","Commercial","Commercial","OpenField","Residential","Residential","Residential"],
        ["OpenField","School","OpenField","Commercial","Hospital","Commercial","OpenField","OpenField","Residential","Residential"],
        ["Residential","Residential","Residential","OpenField","OpenField","OpenField","Commercial","Commercial","OpenField","OpenField"],
        ["Residential","Residential","Residential","Residential","OpenField","Residential","Residential","Commercial","Industrial","OpenField"],
        ["OpenField","Residential","Residential","Residential","School","Residential","Residential","OpenField","OpenField","OpenField"],
        ["Industrial","OpenField","Residential","Residential","Residential","Residential","Residential","Residential","OpenField","Commercial"],
        ["Industrial","Industrial","OpenField","OpenField","Residential","Residential","OpenField","Commercial","Commercial","Commercial"],
    ]

    # raw density values 
    rawDensity=[
        [3842,4242,5185,4716, 648, 521,4500,4700,3200,3400],
        [3200,1297,5220,5100, 834, 521,4900,5000,1500,3100],
        [5185,5100,5100, 648, 521, 648,4800,4600,1297,1000],
        [4900,4716, 648,3842,4242,3900, 648,4500,4300,4100],
        [ 521, 834, 521,3600,1297,3700, 521, 648,4800,4600],
        [5100,4800,4600, 648, 521, 521,3500,3300, 521, 521],
        [5185,5100,4900,4700, 521,4500,4300,3100,1297, 521],
        [ 521,4600,4400,4242, 834,4000,3800, 521, 521, 521],
        [1297, 521,4100,3900,3700,3500,3300,3100, 521,2800],
        [1297,1000, 521, 521,3200,3000, 521,2900,3100,3300],
    ]

    zoneDemand={
        "Residential":4.0,
        "Commercial" :6.0,
        "Industrial" :2.0,
        "Hospital"   :5.0,
        "School"     :3.0,
        "OpenField"  :1.0,
    }

    #thresholds
    lowCut,highCut,_=loadDensityLevels()
    print(f"[Density] Dataset thresholds: Low<{lowCut}, Medium {lowCut}-{highCut}, High>{highCut} persons/sq mi")

    grid=[]
    for r in range(10):
        row=[]
        for c in range(10):
            zone=zMap[r][c]
            lvl=classifyDensity(rawDensity[r][c],lowCut,highCut)  # 1/2/3 label only
            cell=Cell(
                row=r,col=c,
                zone=zone,
                density=lvl,              
                demand=zoneDemand[zone]   
            )
            row.append(cell)
        grid.append(row)

    #hubs
    hubs=[(1,1),(4,4),(7,5),(9,8),(0,7),(3,9),(6,1),(8,3)]
    for r,c in hubs:
        grid[r][c].isHub=True

    #charge pads
    chPads=[(1,2),(0,1),(4,3),(3,4),(7,4),(7,6),(9,7),(9,9),(0,6),(3,8),(6,0),(8,2)]
    for r,c in chPads:
        grid[r][c].isCharging=True

    #Medical
    medPts=[(1,2),(4,3)]
    for r,c in medPts:
        grid[r][c].isMedPickup=True

    #start mein no fly cells
    nfCells=[(2,8),(8,0)]
    for r,c in nfCells:
        grid[r][c].noFly=True

    return grid

def getNeighbors(r:int,c:int,sz:int=10)->List[Tuple[int,int]]:
    
    nbrs=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<sz and 0<=nc<sz:
            nbrs.append((nr,nc))
    return nbrs

def manhattan(a:Tuple[int,int],b:Tuple[int,int])->int:
    
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def printGrid(grid:List[List[Cell]]):
    
    abbrv={"Residential":"RES","Commercial":"COM","Industrial":"IND",
           "Hospital":"HOS","School":"SCH","OpenField":"OPN"}
    lvlChar={1:"L",2:"M",3:"H"}
    for row in grid:
        print(" ".join(f"{abbrv.get(c.zone,'???')}{lvlChar[c.density]}" for c in row))