import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.grid_model import buildGrid, Cell, Drone, Delivery, manhattan
from src.layout_validator import validateLayout
from src.fleet_selector import selectFleet
from src.astar_planner import astar, planDeliveryRoute
from src.delivery_simulator import SimEngine
from src.ml_pipeline import trainDemandModel, trainAnomalyClassifier

st.set_page_config(
    page_title="AeroNet Lite",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Root variables ---- */
:root {
    --beige-50:  #FDFAF5;
    --beige-100: #F5EFE0;
    --beige-200: #EBE0CB;
    --beige-300: #D6C9AF;
    --beige-400: #B8A88A;
    --brown-600: #6B5744;
    --brown-800: #3E2F23;
    --ink:       #1E1A16;
    --muted:     #7A6E64;
    --accent:    #8B6F47;
    --success:   #4A7C59;
    --warning:   #B07D2E;
    --danger:    #923B3B;
    --info:      #3A5F7D;
    --border:    #D6C9AF;
    --card-bg:   #FDFAF5;
    --sidebar-bg:#F0E9DA;
}

/* ---- Global background & font ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--beige-100) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1.5px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

/* ---- Top header strip ---- */
.aeronet-header {
    background: var(--brown-800);
    color: #F5EFE0;
    padding: 18px 32px 14px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: baseline;
    gap: 20px;
    border-bottom: 3px solid var(--beige-300);
}
.aeronet-header .brand {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #F5EFE0 !important;
}
.aeronet-header .sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: var(--beige-300) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ---- Section titles ---- */
.section-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--brown-800);
    border-bottom: 2px solid var(--beige-300);
    padding-bottom: 6px;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}
.section-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: -1rem;
    margin-bottom: 1rem;
}

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(62,47,35,0.07) !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--brown-800) !important;
}

/* ---- Tables / Dataframes ---- */
[data-testid="stDataFrame"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* ---- Buttons ---- */
[data-testid="stButton"] > button {
    background-color: var(--brown-800) !important;
    color: #F5EFE0 !important;
    border: none !important;
    border-radius: 5px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.2rem !important;
    transition: background 0.2s;
}
[data-testid="stButton"] > button:hover {
    background-color: var(--brown-600) !important;
}

/* ---- Sidebar nav radio ---- */
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--ink) !important;
}
[data-testid="stRadio"] > div {
    gap: 4px !important;
}

/* ---- Selectbox & Slider labels ---- */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ---- Alert boxes ---- */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ---- Spinner ---- */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ---- Divider ---- */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ---- Caption / footer ---- */
.sidebar-footer {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: var(--muted);
    text-align: center;
    padding-top: 8px;
}

/* ---- Status badge ---- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-ok  { background: #DDF0E4; color: #2E6B46; border: 1px solid #9BCDB0; }
.badge-err { background: #F8DFDF; color: #7A2525; border: 1px solid #D09090; }
.badge-wrn { background: #FDF2D8; color: #7A5218; border: 1px solid #D4B070; }
.badge-inf { background: #DDE8F0; color: #1E4D6B; border: 1px solid #8AAEC5; }

/* ---- Info label inside ML cards ---- */
.ml-card {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.ml-card h4 {
    font-family: 'Libre Baskerville', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--brown-800);
    margin: 0 0 10px 0;
}

/* ---- Event log entry overrides ---- */
.log-entry {
    padding: 7px 14px;
    border-left: 3px solid var(--border);
    margin-bottom: 5px;
    border-radius: 0 4px 4px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.83rem;
    line-height: 1.5;
    color: var(--ink);
    background: var(--card-bg);
}
.log-entry.err  { border-left-color: var(--danger);  background: #FAF0F0; }
.log-entry.ok   { border-left-color: var(--success); background: #F0F7F2; }
.log-entry.wrn  { border-left-color: var(--warning); background: #FBF5E6; }
.log-entry.info { border-left-color: var(--info);    background: #EDF3F8; }
</style>
""", unsafe_allow_html=True)


ZCLR = {
    "Residential": "#8DAA8A",
    "Commercial":  "#7A9BB5",
    "Industrial":  "#9E9589",
    "Hospital":    "#B57A7A",
    "School":      "#C4A55A",
    "OpenField":   "#C5D4B5",
}

@st.cache_data
def runSim(budget, method):
    grid = buildGrid()
    valRes = validateLayout(grid)
    fleetInfo, drones = selectFleet(grid, budget=budget, method=method)

    demandRes = trainDemandModel()
    anomalyRes = trainAnomalyClassifier()

    sim = SimEngine(grid, drones, fleetInfo)
    def forecaster():
        bestMdl = min(demandRes.items(), key=lambda x: x[1]["rmse"])
        return bestMdl[1]["rmse"]
    sim.run20Steps(forecaster=forecaster)

    routes  = [d.route for d in sim.deliveries if d.route]
    dLabels = [d.assignedDrone for d in sim.deliveries if d.route]

    gridData = []
    for row in grid:
        for c in row:
            gridData.append({
                "row": c.row, "col": c.col, "zone": c.zone, "density": c.density,
                "isHub": c.isHub, "isCharging": c.isCharging,
                "isMedPickup": c.isMedPickup, "noFly": c.noFly, "demand": c.demand
            })

    delData = []
    for d in sim.deliveries:
        delData.append({
            "delId": d.delId, "pickup": str(d.pickup), "dropoff": str(d.dropoff),
            "weight": d.weight, "assignedDrone": d.assignedDrone,
            "status": d.status, "routeCost": d.routeCost
        })

    return {
        "gridData": gridData,
        "valRes": valRes,
        "fleetInfo": fleetInfo,
        "drones": [(d.dId, d.dType, d.cost, d.homeHub, d.battery, d.status) for d in drones],
        "routes": routes,
        "dLabels": dLabels,
        "demandRes": {k: {"mae": v["mae"], "rmse": v["rmse"]} for k, v in demandRes.items()},
        "anomalyRes": {k: {"accuracy": v["accuracy"], "confusionMatrix": v["confusionMatrix"].tolist()} for k, v in anomalyRes.items()},
        "anomalies": sim.anomalies,
        "log": sim.log,
        "delData": delData,
        "summary": {
            "total":     len(sim.deliveries),
            "completed": sum(1 for d in sim.deliveries if d.status == "completed"),
            "inprog":    sum(1 for d in sim.deliveries if d.status == "inprogress"),
            "delayed":   sum(1 for d in sim.deliveries if d.status == "delayed"),
            "failed":    sum(1 for d in sim.deliveries if d.status == "failed"),
            "anomalies": len(sim.anomalies),
            "nofly":     sum(1 for g in gridData if g["noFly"]),
        }
    }

def applyMplTheme():
    plt.rcParams.update({
        "figure.facecolor":  "#FDFAF5",
        "axes.facecolor":    "#FDFAF5",
        "axes.edgecolor":    "#D6C9AF",
        "axes.labelcolor":   "#1E1A16",
        "xtick.color":       "#7A6E64",
        "ytick.color":       "#7A6E64",
        "text.color":        "#1E1A16",
        "axes.titlecolor":   "#3E2F23",
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "font.family":       "serif",
        "grid.color":        "#EBE0CB",
        "grid.alpha":        0.6,
    })


def mkZoneMap(gd):
    applyMplTheme()
    fig, ax = plt.subplots(figsize=(8, 8))
    sz = 10
    abbrv = {"Residential": "RES", "Commercial": "COM", "Industrial": "IND",
              "Hospital": "HOS", "School": "SCH", "OpenField": "OPN"}
    for g in gd:
        r, c = g["row"], g["col"]
        clr = "#3E2F23" if g["noFly"] else ZCLR.get(g["zone"], "#FFF")
        rect = plt.Rectangle((c, sz-1-r), 1, 1, facecolor=clr,
                               edgecolor="#FDFAF5", linewidth=1.5)
        ax.add_patch(rect)
        txt_clr = "#F5EFE0" if g["noFly"] else "#1E1A16"
        ax.text(c+0.5, sz-1-r+0.65, abbrv.get(g["zone"], "?"),
                ha="center", va="center", fontsize=7, fontweight="bold", color=txt_clr)
        marks = []
        if g["isHub"]:      marks.append("H")
        if g["isCharging"]: marks.append("C")
        if g["isMedPickup"]:marks.append("M")
        if g["noFly"]:      marks.append("X")
        if marks:
            ax.text(c+0.5, sz-1-r+0.3, " ".join(marks),
                    ha="center", va="center", fontsize=8,
                    color="#F5EFE0" if g["noFly"] else "#3E2F23", fontweight="bold")
    ax.set_xlim(0, sz); ax.set_ylim(0, sz)
    ax.set_xticks(range(sz)); ax.set_yticks(range(sz))
    ax.set_xticklabels(range(sz)); ax.set_yticklabels(range(sz-1, -1, -1))
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.set_title("City Zone Map")
    ax.set_aspect("equal")
    patches = [mpatches.Patch(color=v, label=k) for k, v in ZCLR.items()]
    patches.append(mpatches.Patch(color="#3E2F23", label="No-Fly"))
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1, 1),
              fontsize=8, framealpha=0.9, edgecolor="#D6C9AF")
    plt.tight_layout()
    return fig

def mkRouteMap(gd, routes, dLabels):
    applyMplTheme()
    fig, ax = plt.subplots(figsize=(8, 8))
    sz = 10
    palette = ["#8B6F47","#4A7C59","#3A5F7D","#923B3B","#6B5744",
               "#B07D2E","#5C7A6E","#7A5B8A","#4E6E8A","#8A6E4E"]
    for g in gd:
        r, c = g["row"], g["col"]
        clr = "#3E2F23" if g["noFly"] else ZCLR.get(g["zone"], "#FFF")
        rect = plt.Rectangle((c, sz-1-r), 1, 1, facecolor=clr,
                               edgecolor="#FDFAF5", linewidth=1, alpha=0.45)
        ax.add_patch(rect)
    for i, route in enumerate(routes):
        if not route: continue
        cl  = palette[i % len(palette)]
        xs  = [c+0.5 for _, c in route]
        ys  = [sz-1-r+0.5 for r, _ in route]
        lbl = dLabels[i] if i < len(dLabels) else f"Route {i+1}"
        ax.plot(xs, ys, color=cl, linewidth=2.2, marker="o",
                markersize=3.5, label=lbl, alpha=0.9)
        ax.plot(xs[0],  ys[0],  marker="s", color=cl, markersize=9,  zorder=5)
        ax.plot(xs[-1], ys[-1], marker="*", color=cl, markersize=12, zorder=5)
    ax.set_xlim(0, sz); ax.set_ylim(0, sz)
    ax.set_xticks(range(sz)); ax.set_yticks(range(sz))
    ax.set_xticklabels(range(sz)); ax.set_yticklabels(range(sz-1, -1, -1))
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.set_title("Drone Route Map")
    ax.set_aspect("equal")
    if routes:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1),
                  fontsize=8, framealpha=0.9, edgecolor="#D6C9AF")
    plt.tight_layout()
    return fig

def mkHeatmap(gd):
    applyMplTheme()
    sz  = 10
    arr = np.zeros((sz, sz))
    for g in gd:
        arr[g["row"]][g["col"]] = g["demand"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(arr, cmap="YlOrBr", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, label="Demand (kg)")
    cbar.ax.yaxis.label.set_color("#1E1A16")
    cbar.ax.tick_params(colors="#7A6E64")
    for r in range(sz):
        for c in range(sz):
            ax.text(c, r, f"{arr[r][c]:.0f}",
                    ha="center", va="center", fontsize=7, color="#1E1A16")
    ax.set_xticks(range(sz)); ax.set_yticks(range(sz))
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.set_title("Demand Heatmap")
    plt.tight_layout()
    return fig


with st.sidebar:
    st.markdown("""
    <div style='padding:16px 4px 10px 4px;'>
        <div style='font-family:"Libre Baskerville",serif;font-size:1.25rem;
                    font-weight:700;color:#3E2F23;letter-spacing:0.02em;'>
            AeroNet Lite
        </div>
        <div style='font-family:"DM Sans",sans-serif;font-size:0.72rem;
                    text-transform:uppercase;letter-spacing:0.12em;
                    color:#7A6E64;margin-top:2px;'>
            Drone Delivery Simulation
        </div>
    </div>
    <hr style='margin:0 0 16px 0;border-color:#D6C9AF;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family:\"DM Sans\",sans-serif;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#7A6E64;margin-bottom:4px;'>Simulation Parameters</div>", unsafe_allow_html=True)

    budget = st.slider("Fleet Budget ($)", 5000, 20000, 10000, 500)
    method = st.selectbox("Fleet Selection Method", ["ga", "brute"], index=0)

    st.markdown("<hr style='margin:16px 0;border-color:#D6C9AF;'>", unsafe_allow_html=True)

    if st.button("Run Simulation", use_container_width=True):
        st.cache_data.clear()

    st.markdown("<hr style='margin:16px 0;border-color:#D6C9AF;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-family:\"DM Sans\",sans-serif;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#7A6E64;margin-bottom:8px;'>Dashboard View</div>", unsafe_allow_html=True)

    view = st.radio(
        label="",
        options=["Overview", "Zone Map", "Route Map", "Demand Heatmap",
                 "ML Results", "Anomaly View", "Event Log"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin:16px 0;border-color:#D6C9AF;'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-footer'>AeroNet Lite v1.0<br>Simulation &mdash; SP2026</div>", unsafe_allow_html=True)


with st.spinner("Running simulation, please wait..."):
    res = runSim(budget, method)


view_subtitles = {
    "Overview":       "Simulation summary, fleet status, and delivery overview",
    "Zone Map":       "10x10 city grid — zone types, hubs, charging pads, medical pickups, no-fly zones",
    "Route Map":      "Drone paths overlaid on the zone map. Square = start, star = end",
    "Demand Heatmap": "Delivery demand intensity per grid cell (kg)",
    "ML Results":     "Demand forecasting regression metrics and anomaly detection classification results",
    "Anomaly View":   "All anomalies flagged during the 20-step simulation",
    "Event Log":      "Step-by-step record of all 20 simulation steps",
}
st.markdown(f"""
<div class='aeronet-header'>
    <span class='brand'>AeroNet Lite</span>
    <span class='sub'>{view}</span>
</div>
""", unsafe_allow_html=True)
st.markdown(f"<div class='section-sub'>{view_subtitles.get(view, '')}</div>", unsafe_allow_html=True)



if view == "Overview":

    
    c1, c2, c3, c4 = st.columns(4)
    s = res["summary"]
    c1.metric("Total Deliveries", s["total"])
    c2.metric("Completed",        s["completed"])
    c3.metric("Delayed",          s["delayed"])
    c4.metric("Failed",           s["failed"])

    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("In Progress", s["inprog"])
    c6.metric("Anomalies",   s["anomalies"])
    c7.metric("No-Fly Cells",s["nofly"])
    c8.metric("Drones",      len(res["drones"]))

    st.markdown("<hr>", unsafe_allow_html=True)

   
    valRes = res["valRes"]
    st.markdown("<div class='section-title'>Layout Validation</div>", unsafe_allow_html=True)
    if not valRes["failed"]:
        passed_str = ", ".join(valRes["passed"])
        st.success(f"Validation passed — Rules: {passed_str}")
    else:
        failed_str = ", ".join(valRes["failed"])
        st.error(f"Validation failed — {failed_str}")
        for e in valRes["errors"][:5]:
            st.warning(e)

    st.markdown("<hr>", unsafe_allow_html=True)

    
    fi = res["fleetInfo"]
    fleet_cost = fi["light"] * 1000 + fi["heavy"] * 1800
    st.markdown("<div class='section-title'>Fleet Configuration</div>", unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Light Drones",  fi["light"])
    fc2.metric("Heavy Drones",  fi["heavy"])
    fc3.metric("Fleet Cost ($)", f"{fleet_cost:,}")
    fc4.metric("Coverage",      f"{fi['covPct']}%")
    st.info(f"Optimisation score: {fi['score']}")

    st.markdown("<hr>", unsafe_allow_html=True)

    
    st.markdown("<div class='section-title'>Deliveries</div>", unsafe_allow_html=True)
    delDf = pd.DataFrame(res["delData"])
    st.dataframe(delDf, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    
    st.markdown("<div class='section-title'>Drone Fleet</div>", unsafe_allow_html=True)
    drnDf = pd.DataFrame(
        res["drones"],
        columns=["ID", "Type", "Cost ($)", "Home Hub", "Battery", "Status"]
    )
    st.dataframe(drnDf, use_container_width=True, hide_index=True)


elif view == "Zone Map":
    fig = mkZoneMap(res["gridData"])
    st.pyplot(fig)
    plt.close(fig)


elif view == "Route Map":
    if res["routes"]:
        fig = mkRouteMap(res["gridData"], res["routes"], res["dLabels"])
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No active routes to display for the current simulation.")


elif view == "Demand Heatmap":
    fig = mkHeatmap(res["gridData"])
    st.pyplot(fig)
    plt.close(fig)


elif view == "ML Results":

    st.markdown("<div class='section-title'>Demand Forecasting — Regression</div>", unsafe_allow_html=True)
    dem_cols = st.columns(max(len(res["demandRes"]), 1))
    for i, (name, metrics) in enumerate(res["demandRes"].items()):
        with dem_cols[i]:
            st.markdown(f"""
            <div class='ml-card'>
                <h4>{name}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("MAE",  metrics["mae"])
            st.metric("RMSE", metrics["rmse"])

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Anomaly Detection — Classification</div>", unsafe_allow_html=True)
    cls_cols = st.columns(max(len(res["anomalyRes"]), 1))
    for i, (name, metrics) in enumerate(res["anomalyRes"].items()):
        with cls_cols[i]:
            st.markdown(f"""
            <div class='ml-card'>
                <h4>{name}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.markdown(
                "<div style='font-size:0.78rem;font-weight:600;text-transform:uppercase;"
                "letter-spacing:0.08em;color:#7A6E64;margin:10px 0 4px 0;'>Confusion Matrix</div>",
                unsafe_allow_html=True
            )
            cmDf = pd.DataFrame(metrics["confusionMatrix"])
            st.dataframe(cmDf, use_container_width=True, hide_index=True)


elif view == "Anomaly View":
    st.markdown("<div class='section-title'>Flagged Anomalies</div>", unsafe_allow_html=True)
    if res["anomalies"]:
        anomDf = pd.DataFrame(res["anomalies"])
        st.dataframe(anomDf, use_container_width=True, hide_index=True)
    else:
        st.info("No anomalies were detected during the simulation run.")


elif view == "Event Log":
    st.markdown("<div class='section-title'>Simulation Event Log</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"DM Sans\",sans-serif;font-size:0.83rem;"
        "color:#7A6E64;margin-bottom:1rem;'>"
        f"{len(res['log'])} events recorded across 20 simulation steps."
        "</div>",
        unsafe_allow_html=True
    )
    for entry in res["log"]:
        entry_lower = entry.lower()
        if "fail" in entry_lower or "delayed" in entry_lower:
            cls = "err"
        elif "completed" in entry_lower:
            cls = "ok"
        elif "rerouted" in entry_lower or "anomaly" in entry_lower:
            cls = "wrn"
        else:
            cls = "info"
        st.markdown(
            f"<div class='log-entry {cls}'>{entry}</div>",
            unsafe_allow_html=True
        )