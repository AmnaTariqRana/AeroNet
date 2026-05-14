import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List,Dict,Tuple,Optional
from src.grid_model import Cell,Drone,Delivery


ZONE_COLORS={
    "Residential":"#4CAF50",
    "Commercial":"#2196F3",
    "Industrial":"#9E9E9E",
    "Hospital":"#F44336",
    "School":"#FF9800",
    "OpenField":"#E8F5E9",
}

def plotZoneMap(grid:List[List[Cell]],title:str="City Zone Map",
               savePath:str=None,show:bool=True):
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    sz=len(grid)

    for r in range(sz):
        for c in range(sz):
            cell=grid[r][c]
            clr=ZONE_COLORS.get(cell.zone,"#FFFFFF")
            if cell.noFly:
                clr="#212121"  
            rect=plt.Rectangle((c,sz-1-r),1,1,facecolor=clr,edgecolor="white",linewidth=1.5)
            ax.add_patch(rect)
            # zone label
            abbrv={"Residential":"RES","Commercial":"COM","Industrial":"IND",
                    "Hospital":"HOS","School":"SCH","OpenField":"OPN"}
            lbl=abbrv.get(cell.zone,"?")
            ax.text(c+0.5,sz-1-r+0.65,lbl,ha="center",va="center",fontsize=7,fontweight="bold")
            markers=[]
            if cell.isHub:markers.append("H")
            if cell.isCharging:markers.append("C")
            if cell.isMedPickup:markers.append("M")
            if cell.noFly:markers.append("X")
            if markers:
                ax.text(c+0.5,sz-1-r+0.3," ".join(markers),ha="center",va="center",
                        fontsize=8,color="white" if cell.noFly else "black",fontweight="bold")

    ax.set_xlim(0,sz);ax.set_ylim(0,sz)
    ax.set_xticks(range(sz));ax.set_yticks(range(sz))
    ax.set_xticklabels(range(sz));ax.set_yticklabels(range(sz-1,-1,-1))
    ax.set_xlabel("Column");ax.set_ylabel("Row")
    ax.set_title(title,fontsize=14,fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True,linewidth=0.3,alpha=0.5)

    
    patches=[mpatches.Patch(color=v,label=k) for k,v in ZONE_COLORS.items()]
    patches.append(mpatches.Patch(color="#212121",label="No-Fly"))
    ax.legend(handles=patches,loc="upper left",bbox_to_anchor=(1,1),fontsize=8)

    plt.tight_layout()
    if savePath:
        fig.savefig(savePath,dpi=150,bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plotRouteMap(grid:List[List[Cell]],routes:List[List[Tuple[int,int]]],
                droneLabels:List[str]=None,title:str="Drone Route Map",
                savePath:str=None,show:bool=True):
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    sz=len(grid)
    colors=["#E91E63","#00BCD4","#FFEB3B","#9C27B0","#FF5722","#3F51B5","#8BC34A","#795548"]

    for r in range(sz):
        for c in range(sz):
            cell=grid[r][c]
            clr=ZONE_COLORS.get(cell.zone,"#FFFFFF")
            if cell.noFly:
                clr="#212121"
            rect=plt.Rectangle((c,sz-1-r),1,1,facecolor=clr,edgecolor="white",
                                linewidth=1,alpha=0.5)
            ax.add_patch(rect)

    
    for i,route in enumerate(routes):
        if not route:
            continue
        clr=colors[i%len(colors)]
        xs=[c+0.5 for _,c in route]
        ys=[sz-1-r+0.5 for r,_ in route]
        lbl=droneLabels[i] if droneLabels and i<len(droneLabels) else f"Route{i+1}"
        ax.plot(xs,ys,color=clr,linewidth=2.5,marker="o",markersize=4,label=lbl,alpha=0.85)
        # start/end markers
        ax.plot(xs[0],ys[0],marker="s",color=clr,markersize=10)
        ax.plot(xs[-1],ys[-1],marker="*",color=clr,markersize=12)

    ax.set_xlim(0,sz);ax.set_ylim(0,sz)
    ax.set_xticks(range(sz));ax.set_yticks(range(sz))
    ax.set_xticklabels(range(sz));ax.set_yticklabels(range(sz-1,-1,-1))
    ax.set_xlabel("Column");ax.set_ylabel("Row")
    ax.set_title(title,fontsize=14,fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper left",bbox_to_anchor=(1,1),fontsize=8)
    plt.tight_layout()
    if savePath:
        fig.savefig(savePath,dpi=150,bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plotDemandHeatmap(grid:List[List[Cell]],title:str="Demand Heatmap",
                      savePath:str=None,show:bool=True):
    sz=len(grid)
    demandArr=np.array([[grid[r][c].demand for c in range(sz)] for r in range(sz)])
    fig,ax=plt.subplots(1,1,figsize=(9,8))
    im=ax.imshow(demandArr,cmap="YlOrRd",interpolation="nearest")
    plt.colorbar(im,ax=ax,label="Demand (kg)")

    for r in range(sz):
        for c in range(sz):
            ax.text(c,r,f"{demandArr[r][c]:.0f}",ha="center",va="center",fontsize=7)

    ax.set_xticks(range(sz));ax.set_yticks(range(sz))
    ax.set_xlabel("Column");ax.set_ylabel("Row")
    ax.set_title(title,fontsize=14,fontweight="bold")
    plt.tight_layout()
    if savePath:
        fig.savefig(savePath,dpi=150,bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plotAnomalyTable(anomalies:List[Dict],title:str="Anomaly Log",
                      savePath:str=None,show:bool=True):
    if not anomalies:
        print("No anomalies recorded.")
        return
    fig,ax=plt.subplots(figsize=(10,max(2,len(anomalies)*0.6+1)))
    ax.axis("off")
    headers=["Drone","Type","Step","Detail"]
    tblData=[[a["drone"],a["type"],str(a["step"]),a.get("detail","")] for a in anomalies]
    tbl=ax.table(cellText=tblData,colLabels=headers,loc="center",cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1,1.5)
    # style header
    for j in range(len(headers)):
        tbl[0,j].set_facecolor("#37474F")
        tbl[0,j].set_text_props(color="white",fontweight="bold")
    ax.set_title(title,fontsize=14,fontweight="bold",pad=20)
    plt.tight_layout()
    if savePath:
        fig.savefig(savePath,dpi=150,bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plotEventLog(log:List[str],title:str="Event Log",
                 savePath:str=None,show:bool=True):
    fig,ax=plt.subplots(figsize=(12,max(4,len(log)*0.25+1)))
    ax.axis("off")
    txt="\n".join(log)
    ax.text(0.02,0.98,txt,transform=ax.transAxes,fontsize=8,verticalalignment="top",
            fontfamily="monospace",wrap=True)
    ax.set_title(title,fontsize=14,fontweight="bold")
    plt.tight_layout()
    if savePath:
        fig.savefig(savePath,dpi=150,bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def saveAllPlots(grid,routes,droneLabels,anomalies,log,outDir="report/figures"):
    import os
    os.makedirs(outDir,exist_ok=True)
    plotZoneMap(grid,savePath=f"{outDir}/zone_map.png",show=False)
    plotRouteMap(grid,routes,droneLabels,savePath=f"{outDir}/route_map.png",show=False)
    plotDemandHeatmap(grid,savePath=f"{outDir}/demand_heatmap.png",show=False)
    plotAnomalyTable(anomalies,savePath=f"{outDir}/anomaly_log.png",show=False)
    plotEventLog(log,savePath=f"{outDir}/event_log.png",show=False)
    print(f"[INFO] All plots saved to {outDir}/")