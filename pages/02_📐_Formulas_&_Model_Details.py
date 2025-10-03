# 📐 Formulas & Model Details
# Save as: pages/02_📐_Formulas_&_Model_Details.py
#
# Shows the math and a live snapshot of your current Base parameters along with the fitted
# Intuition model parameters (if you've run a fit in this session).

import streamlit as st

st.set_page_config(page_title="Formulas — Headroom Model", layout="wide")
st.title("📐 Formulas & Model Details")
st.caption("Mathematical definitions and the current **Base** vs **Intuition** model parameters. Development awards **no points** for ΔP ≤ 0.")

# Pull Base from UI keys (main page initializes these)
base = None
if all(k in st.session_state for k in [
    "ui_max_perf","ui_alpha_perf","ui_low_pct_perf","ui_high_pct_perf",
    "ui_max_dev","ui_alpha_dev","ui_delta","ui_low_pct_dev","ui_high_pct_dev"
]):
    base = {
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

fit = st.session_state.get("fitted_model")

st.markdown("---")

# =============== Z-bounds and mapping ===============
st.header("1) Z-bounds and z-score mapping")

st.latex(r"""
\begin{aligned}
Z_{\min} &= \Phi^{-1}\!\left(\frac{\text{low\_pct}}{100}\right), \\
Z_{\max} &= \Phi^{-1}\!\left(\frac{\text{high\_pct}}{100}\right),
\qquad Z_{\max} > Z_{\min}.
\end{aligned}
""")

st.latex(r"""
\text{Given } P \in [0,100],\quad
\tilde P=\operatorname{clip}\!\left(\frac{P}{100},\varepsilon,1-\varepsilon\right),\quad
Z=\operatorname{clip}\!\big(\Phi^{-1}(\tilde P),\,Z_{\min},\,Z_{\max}\big).
""")

st.caption("Here, \(\\Phi^{-1}\) is the inverse CDF of the standard normal; ε is a tiny constant (e.g., 1e−6).")

st.markdown("---")

# =============== Performance points ===============
st.header("2) Performance Points")

st.latex(r"""
\begin{aligned}
\text{frac}&=\operatorname{clip}\!\left(\frac{Z-Z_{\min}}{Z_{\max}-Z_{\min}},\,0,\,1\right),\\
\text{PerfPts}&=\text{Max}_{perf}\cdot \text{frac}^{\,\alpha_{perf}},\qquad
\alpha_{perf}>0,\ \text{Max}_{perf}>0.
\end{aligned}
""")

st.markdown("""
- \(\\alpha_{perf} < 1\): relatively more points in the middle percentiles.  
- \(\\alpha_{perf} > 1\): concentrates points near the very top.
""")

st.markdown("---")

# =============== Development points (improvement only) ===============
st.header("3) Development Points — improvement only (ΔP ≤ 0 ⇒ 0)")

st.latex(r"""
\text{Let } Z_{prev}=\operatorname{z}(P_{prev}),\ Z_{curr}=\operatorname{z}(P_{curr}),\
b=\left(\frac{P_{prev}}{100}\right)^{\delta}.
""")

st.latex(r"""
\begin{aligned}
\Delta P&=P_{curr}-P_{prev}\le 0 \Rightarrow \text{DevPts}=0.\\
\Delta P&>0:\ \text{rel}_{+}=\operatorname{clip}\!\left(\frac{Z_{curr}-Z_{prev}}{Z_{\max}-Z_{prev}},0,1\right),\\
&\qquad\text{DevPts}=\text{Max}_{dev}\cdot b\cdot (\text{rel}_{+})^{\alpha_{dev}},\ \ \text{clipped to }[0,\text{Max}_{dev}].
\end{aligned}
""")

st.markdown("""
- \(\\alpha_{dev}\) shapes how improvements escalate (power curve).  
- \(\\delta\) adjusts by starting level: \(\\delta>0\) favors higher starters; \(\\delta<0\) favors lower starters.
""")

st.markdown("---")

# =============== Parameters snapshot ===============
st.header("4) Parameters snapshot")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Base model**")
    if base:
        st.write({
            "Max_perf": base["max_perf"], "α_perf": base["alpha_perf"],
            "Max_dev": base["max_dev"], "α_dev": base["alpha_dev"], "δ": base["delta"],
            "Perf Z (%):": (base["low_pct_perf"], base["high_pct_perf"]),
            "Dev Z (%):":  (base["low_pct_dev"],  base["high_pct_dev"]),
        })
    else:
        st.info("Open the main page to initialize Base model parameters.")
with c2:
    st.markdown("**Intuition model**")
    if fit:
        st.write({
            "Max_perf": fit["max_perf"], "α_perf": fit["alpha_perf"],
            "Max_dev": fit["max_dev"], "α_dev": fit["alpha_dev"], "δ": fit["delta"],
            "Perf Z (%):": (fit["low_pct_perf"], fit["high_pct_perf"]),
            "Dev Z (%):":  (fit["low_pct_dev"], fit["high_pct_dev"]),
            "Fit": f"{fit['success']} — {fit['message']}",
            "When": fit["timestamp"],
        })
    else:
        st.info("No Intuition model fitted yet.")

st.markdown("---")

st.header("5) Notes")
st.markdown("""
- The **Intuition fit** solves a bounded least-squares problem for  
  \\(\\{\\text{Max}_{perf},\\ \\alpha_{perf},\\ \\text{Max}_{dev},\\ \\alpha_{dev},\\ \\delta\\}\\), with **z-bounds fixed**.
- Only benchmark rows where you entered **Intuition** points are used. Development uses only rows with **ΔP>0**.
- Development negatives are not penalized; they simply receive 0 points.
""")
