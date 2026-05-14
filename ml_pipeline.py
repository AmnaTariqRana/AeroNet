import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                              accuracy_score,confusion_matrix,classification_report)



def genDemandData(nSamples:int=500)->pd.DataFrame:
    
    np.random.seed(42)
    hr=np.random.randint(0,24,nSamples)
    dow=np.random.randint(0,7,nSamples)
    temp=np.random.uniform(5,40,nSamples)
    weather=np.random.choice([1,2,3,4],nSamples,p=[0.5,0.3,0.15,0.05])  # 1=clear,4=storm
    zoneDens=np.random.choice([1000,3000,5000,7000],nSamples)

    
    demand=(0.8*hr+0.5*dow+0.3*temp-2.0*weather+0.002*zoneDens
            +np.random.normal(0,3,nSamples))
    demand=np.clip(demand,0,None)

    df=pd.DataFrame({"hour":hr,"dayOfWeek":dow,"temperature":temp,
                      "weather":weather,"zoneDensity":zoneDens,"demand":demand})
    return df

def loadBikeSharingData(path:str)->pd.DataFrame:
    if os.path.exists(path):
        df=pd.read_csv(path)
        #relevant columns
        cols=["season","holiday","workingday","weather","temp","humidity","windspeed","count"]
        avail=[c for c in cols if c in df.columns]
        df=df[avail].dropna()
        if "count" in df.columns:
            df=df.rename(columns={"count":"demand"})
        return df
    return None

def trainDemandModel(df:pd.DataFrame=None,usePath:str=None)->dict:
    
    if usePath:
        df=loadBikeSharingData(usePath)
    if df is None:
        df=genDemandData()
        print("[INFO] Using synthetic demand data (Bike Sharing proxy)")

    
    target="demand"
    feats=[c for c in df.columns if c!=target]
    X=np.array(df[feats].values,dtype=float)
    y=np.array(df[target].values,dtype=float)

    xTr,xTe,yTr,yTe=train_test_split(X,y,test_size=0.2,random_state=42)

    results={}

    lr=LinearRegression()
    lr.fit(xTr,yTr)
    yPredLr=lr.predict(xTe)
    maeLr=mean_absolute_error(yTe,yPredLr)
    rmseLr=np.sqrt(mean_squared_error(yTe,yPredLr))
    results["LinearRegression"]={"mae":round(maeLr,3),"rmse":round(rmseLr,3),"model":lr}

    rfr=RandomForestRegressor(n_estimators=100,random_state=42,max_depth=10)
    rfr.fit(xTr,yTr)
    yPredRf=rfr.predict(xTe)
    maeRf=mean_absolute_error(yTe,yPredRf)
    rmseRf=np.sqrt(mean_squared_error(yTe,yPredRf))
    results["RandomForestRegressor"]={"mae":round(maeRf,3),"rmse":round(rmseRf,3),"model":rfr}

    
    
    print("DEMAND FORECASTING RESULTS")
    print(f"Dataset: {len(df)} samples, {len(feats)} features")
    print(f"Features: {feats}")
    print(f"Train/Test split: 80/20\n")
    for name,r in results.items():
        print(f"  {name}:")
        print(f"    MAE  = {r['mae']}")
        print(f"    RMSE = {r['rmse']}")
    bestModel=min(results.items(),key=lambda x:x[1]["rmse"])
    print(f"\nBest model: {bestModel[0]} (RMSE={bestModel[1]['rmse']})")

    return results



def genAnomalyData(nSamples:int=600)->pd.DataFrame:
    np.random.seed(42)
    data=[]
    labels=["Normal","BatteryAnomaly","RouteAnomaly","SensorSpike"]
    perClass=nSamples//4

    for i in range(perClass):
        data.append({"batteryDrop":np.random.uniform(1,8),
                      "speed":np.random.uniform(8,15),
                      "routeDeviation":np.random.uniform(0,4),
                      "altChange":np.random.uniform(0,5),
                      "speedChange":np.random.uniform(0,4),
                      "label":"Normal"})
    for i in range(perClass):
        data.append({"batteryDrop":np.random.uniform(12,40),
                      "speed":np.random.uniform(5,14),
                      "routeDeviation":np.random.uniform(0,5),
                      "altChange":np.random.uniform(0,6),
                      "speedChange":np.random.uniform(0,5),
                      "label":"BatteryAnomaly"})
    for i in range(perClass):
        data.append({"batteryDrop":np.random.uniform(1,10),
                      "speed":np.random.uniform(5,13),
                      "routeDeviation":np.random.uniform(6,20),
                      "altChange":np.random.uniform(0,7),
                      "speedChange":np.random.uniform(0,5),
                      "label":"RouteAnomaly"})
    for i in range(perClass):
        data.append({"batteryDrop":np.random.uniform(1,9),
                      "speed":np.random.uniform(7,16),
                      "routeDeviation":np.random.uniform(0,5),
                      "altChange":np.random.uniform(10,35),
                      "speedChange":np.random.uniform(8,25),
                      "label":"SensorSpike"})

    df=pd.DataFrame(data)
    return df.sample(frac=1,random_state=42).reset_index(drop=True)

def trainAnomalyClassifier(df:pd.DataFrame=None)->dict:
    if df is None:
        df=genAnomalyData()
        print("[INFO] Using synthetic anomaly data")

    feats=[c for c in df.columns if c!="label"]
    X=np.array(df[feats].values,dtype=float)
    y=np.array(df["label"].values)

    xTr,xTe,yTr,yTe=train_test_split(X,y,test_size=0.2,random_state=42)

    models={
        "DecisionTree":DecisionTreeClassifier(max_depth=8,random_state=42),
        "RandomForest":RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42),
        "KNN":KNeighborsClassifier(n_neighbors=5),
    }

    results={}
    print("ANOMALY DETECTION RESULTS")
    print(f"Dataset: {len(df)} samples, {len(feats)} features")
    print(f"Features: {feats}")
    print(f"Classes: {list(df['label'].unique())}")
    print(f"Train/Test split: 80/20\n")

    for name,mdl in models.items():
        mdl.fit(xTr,yTr)
        yPred=mdl.predict(xTe)
        acc=accuracy_score(yTe,yPred)
        cm=confusion_matrix(yTe,yPred)
        results[name]={"accuracy":round(acc,4),"confusionMatrix":cm,"model":mdl}

        print(f"  {name}:")
        print(f"    Accuracy = {acc:.4f}")
        print(f"    Confusion Matrix:")
        for row in cm:
            print(f"      {row}")
        print()

    bestModel=max(results.items(),key=lambda x:x[1]["accuracy"])
    print(f"Best model: {bestModel[0]} (Accuracy={bestModel[1]['accuracy']})")
    print("="*60)

    return results

def runFullPipeline():
    demandRes=trainDemandModel()
    anomalyRes=trainAnomalyClassifier()
    return demandRes,anomalyRes
