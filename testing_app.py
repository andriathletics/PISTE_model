# Headroom Model — Base vs Intuition Fit
# Save as: app.py
#
# pip install streamlit scipy numpy matplotlib pandas

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import least_squares
from datetime import datetime

st.set_page_config(page_title="Headroom Model — Base vs Intuition", layout="wide")

# -----------------------------------------------------------------------------
# Session defaults (match original shape; only change: max points = 100)
# Widgets use key-only pattern (no value=) to avoid "two sources of truth" warnings.
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    # Performance (original defaults, except Max=100)
    ui_max_perf=100.0,       # was 50 → now 100
    ui_alpha_perf=1.0,
    ui_low_pct_perf=0.1,
    ui_high_pct_perf=99.9,

    # Development (original defaults, except Max=100)
    ui_max_dev=100.0,        # was 50 → now 100
    ui_alpha_dev=1.0,
    ui_delta=0.0,
    ui_low_pct_dev=0.1,
    ui_high_pct_dev=99.9,

    # Benchmark grid presets (edit if you want other steps)
    starts_list=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0],
    deltas_list=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0],
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Optional: quick reset to the defaults above
with st.sidebar:
    if st.button("↩️ Reset model to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.session_state.pop("fitted_model", None)
        st.rerun()

# Utility: read current Base params from UI state
def base_params_from_ui():
    return {
        "max_perf": st.session_state["ui_max_perf"],
        "alpha_perf": st.session_state["ui_alpha_perf"],
        "low_pct_perf": st.session_state["ui_low_pct_perf"],
        "high_pct_perf": st.session_state["ui_high_pct_perf"],
        "max_dev": st.session_state["ui_max_dev"],
        "alpha_dev": st.session_state["ui_alpha_dev"],
        "delta": st.session_state["ui_delta"],
        "low_pct_dev": st.session_state["ui_low_pct_dev"],
        "high_pct_dev": st.session_state["ui_high_pct_dev"],
    }

# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def z_bounds(low_pct, high_pct):
    low = np.clip(low_pct / 100.0, 1e-6, 1 - 1e-6)
    high = np.clip(high_pct / 100.0, 1e-6, 1 - 1e-6)
    zmin = norm.ppf(low); zmax = norm.ppf(high)
    if zmax <= zmin:
        zmax = zmin + 1e-12
    return zmin, zmax

def z_from_percentile(p, zmin, zmax):
    p = np.clip(np.asarray(p) / 100.0, 1e-6, 1 - 1e-6)
    z = norm.ppf(p)
    return np.clip(z, zmin, zmax)

def performance_points(p_curr, max_perf, alpha_perf, zmin, zmax):
    z = z_from_percentile(p_curr, zmin, zmax)
    frac = np.clip((z - zmin) / (zmax - zmin), 0, 1)
    pts = max_perf * (frac ** alpha_perf)
    return np.clip(pts, 0, max_perf)

# Development: improvement only (ΔP ≤ 0 ⇒ 0 points)
def development_points(p_prev, p_curr, max_dev, alpha_dev, delta, zmin, zmax):
    p_prev = np.asarray(p_prev, dtype=float)
    p_curr = np.asarray(p_curr, dtype=float)
    Zprev = z_from_percentile(p_prev, zmin, zmax)
    Zcurr = z_from_percentile(p_curr, zmin, zmax)
    pts = np.zeros_like(np.atleast_1d(p_curr), dtype=float)
    base = (np.clip(p_prev, 0, 100) / 100.0) ** delta
    mask_pos = (p_curr > p_prev)
    if np.any(mask_pos):
        denom = np.maximum(np.asarray(zmax) - Zprev, 1e-12)
        rel = np.clip((Zcurr - Zprev) / denom, 0.0, 1.0)
        pts = np.where(mask_pos, max_dev * base * (rel ** alpha_dev), pts)
    return np.clip(pts, 0.0, max_dev)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Headroom Model — Base vs Intuition Fit")
st.caption("Step 1: set a **Base model**. Step 2: add your **Intuition** on a benchmark grid. Step 3: **Fit** a new model to match your intuition and compare both.")

# -----------------------------------------------------------------------------
# STEP 1 — BASE MODEL (clear sections with short explanations)
# -----------------------------------------------------------------------------
st.markdown("## 1) Set the Base Model")

with st.container(border=True):
    st.subheader("Performance parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Max performance points", min_value=1.0, step=1.0, key="ui_max_perf",
            help="Sets the **ceiling** of Performance points at the 100th percentile."
        )
    with c2:
        st.slider(
            "α_perf (nonlinearity)", min_value=0.10, max_value=3.00, step=0.05, key="ui_alpha_perf",
            help="Shapes the Performance curve on z-scale headroom. **<1**: more points mid-range. **>1**: more points near the top."
        )
    with c3:
        st.number_input(
            "Perf Z lower bound (percentile)", min_value=0.0, max_value=10.0, step=0.1, key="ui_low_pct_perf",
            help="Floor percentile for z-scores (prevents infinities and defines the lower headroom bound)."
        )
        st.number_input(
            "Perf Z upper bound (percentile)", min_value=90.0, max_value=100.0, step=0.1, key="ui_high_pct_perf",
            help="Ceiling percentile for z-scores (prevents infinities and defines the upper headroom bound)."
        )

with st.container(border=True):
    st.subheader("Development parameters (ΔP ≤ 0 ⇒ 0 pts)")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.number_input(
            "Max development points", min_value=1.0, step=1.0, key="ui_max_dev",
            help="Sets the **maximum** Development award for a large positive improvement."
        )
    with c5:
        st.slider(
            "α_dev (improvement curve)", min_value=0.10, max_value=3.00, step=0.05, key="ui_alpha_dev",
            help="Shapes how Development grows with relative improvement. **<1**: gentler; **>1**: emphasizes big gains."
        )
    with c6:
        st.slider(
            "δ (baseline weighting)", min_value=-3.00, max_value=3.00, step=0.05, key="ui_delta",
            help="Adjusts reward by starting level. **δ>0** favors higher starters; **δ<0** favors lower starters."
        )
        st.number_input(
            "Dev Z lower bound (percentile)", min_value=0.0, max_value=10.0, step=0.1, key="ui_low_pct_dev",
            help="Floor percentile for Development z-scores."
        )
        st.number_input(
            "Dev Z upper bound (percentile)", min_value=90.0, max_value=100.0, step=0.1, key="ui_high_pct_dev",
            help="Ceiling percentile for Development z-scores."
        )

base = base_params_from_ui()
zmin_perf, zmax_perf = z_bounds(base["low_pct_perf"], base["high_pct_perf"])
zmin_dev,  zmax_dev  = z_bounds(base["low_pct_dev"],  base["high_pct_dev"])

# -----------------------------------------------------------------------------
# Base model curves (quick visual)
# -----------------------------------------------------------------------------
st.markdown("### Base model curves")

# Performance curve
x_perf = np.linspace(0, 100, 600)
y_perf_base = performance_points(x_perf, base["max_perf"], base["alpha_perf"], zmin_perf, zmax_perf)
figp, axp = plt.subplots(figsize=(6.5, 4))
axp.plot(x_perf, y_perf_base, label="Base")
axp.set_xlabel("Current percentile")
axp.set_ylabel("Performance Points")
axp.grid(True, linestyle="--", alpha=0.6)
axp.legend()
st.pyplot(figp)

# Development traces (positive ΔP only)
starts_preview = np.arange(50.0, 100.0 + 1e-9, 10.0)
delta_range = np.arange(0.0, 20.0 + 1e-9, 1.0)  # includes 0 (→ 0 pts)
figd, axd = plt.subplots(figsize=(7.5, 4.5))
for p0 in starts_preview:
    p_currs = np.clip(p0 + delta_range, 0.0, 100.0)
    y = development_points(p0, p_currs, base["max_dev"], base["alpha_dev"], base["delta"], zmin_dev, zmax_dev)
    axd.plot(delta_range, y, label=f"Start {p0:.0f}")
axd.axhline(0, linewidth=1)
axd.set_xlim(left=0)
axd.set_xlabel("ΔP (percentile points)")
axd.set_ylabel("Development Points")
axd.grid(True, linestyle="--", alpha=0.6)
axd.legend(ncol=2, fontsize=9, frameon=False)
st.pyplot(figd)

st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 2 — Benchmark grid (pre-filled variations) + Intuition targets
# -----------------------------------------------------------------------------
st.markdown("## 2) Benchmark grid & your intuition")

def make_benchmark_grid(starts, deltas):
    rows = []
    for p0 in starts:
        for d in deltas:
            p1 = min(100.0, p0 + d)
            if d <= 0:
                continue  # dev only cares about improvements
            rows.append({"Start (P_prev)": float(p0), "ΔP": float(d), "Current (P_curr)": float(p1)})
    return pd.DataFrame(rows)

bench_df = make_benchmark_grid(st.session_state["starts_list"], st.session_state["deltas_list"])

# Persist intuition entries across reruns
if "intuition_targets" not in st.session_state:
    st.session_state["intuition_targets"] = pd.DataFrame(
        columns=["Start (P_prev)","ΔP","PerfPts (intuition)","DevPts (intuition)"]
    )

# Merge any previous intuition onto the current grid
bench_df = bench_df.merge(
    st.session_state["intuition_targets"],
    on=["Start (P_prev)","ΔP"], how="left"
)

# Compute Base model points for the grid
bench_df["PerfPts (Base)"] = np.round(
    performance_points(bench_df["Current (P_curr)"], base["max_perf"], base["alpha_perf"], zmin_perf, zmax_perf), 3
)
bench_df["DevPts (Base)"] = np.round(
    development_points(bench_df["Start (P_prev)"], bench_df["Current (P_curr)"], base["max_dev"], base["alpha_dev"], base["delta"], zmin_dev, zmax_dev), 3
)

st.caption("Base model points are precomputed. Type **Intuition** points only where you want the fit to follow your judgement; blanks are ignored.")
bench_edited = st.data_editor(
    bench_df[[
        "Start (P_prev)","ΔP","Current (P_curr)",
        "PerfPts (Base)","DevPts (Base)",
        "PerfPts (intuition)","DevPts (intuition)"
    ]],
    use_container_width=True, hide_index=True, num_rows="dynamic",
    column_config={
        "PerfPts (Base)": st.column_config.NumberColumn("PerfPts (Base)", disabled=True),
        "DevPts (Base)":  st.column_config.NumberColumn("DevPts (Base)", disabled=True),
        "PerfPts (intuition)": st.column_config.NumberColumn("PerfPts (intuition)", step=0.001, help="Your desired Performance points."),
        "DevPts (intuition)":  st.column_config.NumberColumn("DevPts (intuition)",  step=0.001, help="Your desired Development points (ΔP>0 only)."),
    },
    key="bench_editor",
)

# Persist intuition edits
st.session_state["intuition_targets"] = bench_edited[["Start (P_prev)","ΔP","PerfPts (intuition)","DevPts (intuition)"]].copy()

st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 3 — Fit an Intuition model to your edited targets
# -----------------------------------------------------------------------------
st.markdown("## 3) Fit Intuition model to your targets")

cL, cR = st.columns([3,1])
with cL:
    use_perf = st.checkbox("Use PerfPts (intuition) targets", value=True)
    use_dev  = st.checkbox("Use DevPts (intuition) targets", value=True)
    st.caption("Only rows with Intuition values are included. Development uses only ΔP>0 rows.")
with cR:
    w_perf = st.slider("Perf residual weight", 0.0, 5.0, 1.0, 0.1,
                       help="Relative weight of Performance errors in the least-squares fit.")
    w_dev  = st.slider("Dev residual weight",  0.0, 5.0, 1.0, 0.1,
                       help="Relative weight of Development errors in the least-squares fit.")

# ----- Build masks SAFELY (Series logic) -----
perf_tgt = pd.to_numeric(bench_edited["PerfPts (intuition)"], errors="coerce")
dev_tgt  = pd.to_numeric(bench_edited["DevPts (intuition)"],  errors="coerce")

p_prev = bench_edited["Start (P_prev)"].astype(float)
p_curr = bench_edited["Current (P_curr)"].astype(float)

# Start from element-wise conditions
perf_mask = perf_tgt.notna()                    # rows where Perf intuition provided
dev_mask  = dev_tgt.notna() & (p_curr > p_prev) # rows where Dev intuition provided and ΔP>0

# Respect the toggles
if not use_perf:
    perf_mask = pd.Series(False, index=bench_edited.index)
if not use_dev:
    dev_mask  = pd.Series(False, index=bench_edited.index)

n_perf = int(perf_mask.sum())
n_dev  = int(dev_mask.sum())
st.caption(f"Targets included — Performance: **{n_perf}**, Development: **{n_dev}**")

def pack(mp, ap, md, ad, dlt): return np.array([mp, ap, md, ad, dlt], dtype=float)
def unpack(x): return x[0], x[1], x[2], x[3], x[4]
lb = np.array([1.0, 0.10, 1.0, 0.10, -3.0])
ub = np.array([500.0, 3.00, 500.0, 3.00,  3.0])

x0 = pack(base["max_perf"], base["alpha_perf"], base["max_dev"], base["alpha_dev"], base["delta"])

# Only compute residuals inside the button click
fit_disabled = (n_perf == 0 and n_dev == 0) or (w_perf == 0 and w_dev == 0)
if st.button("Fit Intuition model", type="primary", disabled=fit_disabled):

    def residuals(x):
        mp, ap, md, ad, dlt = unpack(x)
        res = []
        if n_perf > 0 and w_perf > 0:
            y_hat = performance_points(p_curr[perf_mask], mp, ap, zmin_perf, zmax_perf)
            y_tgt = perf_tgt[perf_mask].values.astype(float)
            res.append(w_perf * (y_hat - y_tgt))
        if n_dev > 0 and w_dev > 0:
            y_hat = development_points(p_prev[dev_mask], p_curr[dev_mask], md, ad, dlt, zmin_dev, zmax_dev)
            y_tgt = dev_tgt[dev_mask].values.astype(float)
            res.append(w_dev * (y_hat - y_tgt))
        if not res:
            return np.zeros(1)
        return np.concatenate(res)

    sol = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
    mp, ap, md, ad, dlt = unpack(sol.x)

    # Save fitted model (do not overwrite base sliders; we compare below)
    fitted = {
        "max_perf": float(mp), "alpha_perf": float(ap),
        "low_pct_perf": base["low_pct_perf"], "high_pct_perf": base["high_pct_perf"],
        "max_dev": float(md), "alpha_dev": float(ad), "delta": float(dlt),
        "low_pct_dev": base["low_pct_dev"], "high_pct_dev": base["high_pct_dev"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": bool(sol.success), "message": str(sol.message),
    }
    st.session_state["fitted_model"] = fitted

# -----------------------------------------------------------------------------
# STEP 4 — Compare Base vs Intuition model
# -----------------------------------------------------------------------------
st.markdown("## 4) Compare Base vs Intuition")

fitted = st.session_state.get("fitted_model")
cA, cB = st.columns(2)
with cA:
    st.markdown("**Base model parameters**")
    st.write({
        "Max_perf": base["max_perf"], "α_perf": base["alpha_perf"],
        "Max_dev": base["max_dev"], "α_dev": base["alpha_dev"], "δ": base["delta"],
        "Perf Z (%):": (base["low_pct_perf"], base["high_pct_perf"]),
        "Dev Z (%):":  (base["low_pct_dev"],  base["high_pct_dev"]),
    })
with cB:
    st.markdown("**Intuition model parameters**")
    if fitted:
        st.write({
            "Max_perf": fitted["max_perf"], "α_perf": fitted["alpha_perf"],
            "Max_dev": fitted["max_dev"], "α_dev": fitted["alpha_dev"], "δ": fitted["delta"],
            "Perf Z (%):": (fitted["low_pct_perf"], fitted["high_pct_perf"]),
            "Dev Z (%):":  (fitted["low_pct_dev"],  fitted["high_pct_dev"]),
            "Fit": f"{fitted['success']} — {fitted['message']}",
            "When": fitted["timestamp"],
        })
        st.caption("Z-bounds are held fixed during fitting.")
    else:
        st.info("No Intuition model yet. Enter intuition targets and click **Fit Intuition model**.")

# Curves overlay
st.markdown("### Curves overlay")

# Performance overlay
figP, axP = plt.subplots(figsize=(6.5, 4))
axP.plot(x_perf, y_perf_base, label="Base")
if fitted:
    zminP2, zmaxP2 = z_bounds(fitted["low_pct_perf"], fitted["high_pct_perf"])
    y_perf_fit = performance_points(x_perf, fitted["max_perf"], fitted["alpha_perf"], zminP2, zmaxP2)
    axP.plot(x_perf, y_perf_fit, linestyle="--", label="Intuition")
axP.set_xlabel("Current percentile")
axP.set_ylabel("Performance Points")
axP.grid(True, linestyle="--", alpha=0.6)
axP.legend()
st.pyplot(figP)

# Development overlay
figD2, axD2 = plt.subplots(figsize=(7.5, 4.5))
for p0 in starts_preview:
    p_currs = np.clip(p0 + delta_range, 0.0, 100.0)
    y_base = development_points(p0, p_currs, base["max_dev"], base["alpha_dev"], base["delta"], zmin_dev, zmax_dev)
    axD2.plot(delta_range, y_base, label=f"Base start {p0:.0f}")
    if fitted:
        zminD2, zmaxD2 = z_bounds(fitted["low_pct_dev"], fitted["high_pct_dev"])
        y_fit = development_points(p0, p_currs, fitted["max_dev"], fitted["alpha_dev"], fitted["delta"], zminD2, zmaxD2)
        axD2.plot(delta_range, y_fit, linestyle="--", label=f"Intuition start {p0:.0f}")
axD2.axhline(0, linewidth=1)
axD2.set_xlim(left=0)
axD2.set_xlabel("ΔP (percentile points)")
axD2.set_ylabel("Development Points")
axD2.grid(True, linestyle="--", alpha=0.6)
axD2.legend(ncol=2, fontsize=9, frameon=False)
st.pyplot(figD2)

# Benchmark grid scored under both models
st.markdown("### Benchmark grid — Base vs Intuition scoring")
compare_df = bench_edited.copy()
# Base already computed in bench_df/bench_edited
if fitted:
    zminP2, zmaxP2 = z_bounds(fitted["low_pct_perf"], fitted["high_pct_perf"])
    zminD2, zmaxD2 = z_bounds(fitted["low_pct_dev"],  fitted["high_pct_dev"])
    compare_df["PerfPts (Intuition model)"] = np.round(
        performance_points(compare_df["Current (P_curr)"], fitted["max_perf"], fitted["alpha_perf"], zminP2, zmaxP2), 3
    )
    compare_df["DevPts (Intuition model)"] = np.round(
        development_points(compare_df["Start (P_prev)"], compare_df["Current (P_curr)"], fitted["max_dev"], fitted["alpha_dev"], fitted["delta"], zminD2, zmaxD2), 3
    )
    compare_df["Δ Perf (Intuition-Base)"] = np.round(compare_df["PerfPts (Intuition model)"] - compare_df["PerfPts (Base)"], 3)
    compare_df["Δ Dev (Intuition-Base)"]  = np.round(compare_df["DevPts (Intuition model)"]  - compare_df["DevPts (Base)"], 3)

display_cols = [
    "Start (P_prev)","ΔP","Current (P_curr)",
    "PerfPts (Base)","DevPts (Base)",
    "PerfPts (intuition)","DevPts (intuition)"
]
if fitted:
    display_cols += ["PerfPts (Intuition model)","DevPts (Intuition model)","Δ Perf (Intuition-Base)","Δ Dev (Intuition-Base)"]

st.dataframe(compare_df[display_cols], use_container_width=True, hide_index=True)

