"""
1D Beam FEM Simulator — TgN Virtual Lab
Streamlit app implementing Euler-Bernoulli beam FEM with Hermitian cubic shape functions.
Author: TgN (finiteelementsimulationsbytgn.blogspot.com)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import io

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="1D Beam FEM Simulator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS  (clean white, matching blog style)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans Pro', Arial, sans-serif; }
h1 { color: #3d77ca; border-bottom: 2px solid #91b3ed; padding-bottom: 6px; }
h2 { color: #3d77ca; }
h3 { color: #3d77ca; }
.stTabs [data-baseweb="tab-list"] { background: #91b3ed; border-radius: 4px 4px 0 0; }
.stTabs [data-baseweb="tab"] { color: #265fd3; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #f1f1f1 !important; color: #000 !important; }
.stDataFrame { border: 1px solid #e0e8f0; }
div[data-testid="stMetricValue"] { color: #3d77ca; font-family: 'Courier New', monospace; }
.sidebar-note { background:#f0f4fa; border-left:3px solid #91b3ed;
                padding:8px 10px; font-size:12px; color:#555; margin-bottom:8px; border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SECTION PROPERTIES
# ─────────────────────────────────────────
def calc_section(profile, b, h, t=None):
    """Return (I, W, y_max) for given profile."""
    if profile == "Rectangle":
        I = b * h**3 / 12
        W = b * h**2 / 6
        y_max = h / 2
    elif profile == "Circle":
        r = b / 2
        I = np.pi * r**4 / 4
        W = np.pi * r**3 / 4
        y_max = r
    elif profile == "I-Beam":
        tf = t if t else h * 0.15
        tw = max(b * 0.1, 2.0)
        hw = h - 2 * tf
        I = (b * h**3 / 12) - ((b - tw) * hw**3 / 12)
        W = I / (h / 2)
        y_max = h / 2
    elif profile == "Hollow Rectangle":
        t_ = t if t else min(b, h) * 0.1
        bi, hi = b - 2*t_, h - 2*t_
        I = (b * h**3 - bi * hi**3) / 12
        W = I / (h / 2)
        y_max = h / 2
    else:
        I = b * h**3 / 12
        W = b * h**2 / 6
        y_max = h / 2
    return float(I), float(W), float(y_max)

# ─────────────────────────────────────────
# BEAM TYPE PRESETS
# ─────────────────────────────────────────
BEAM_TYPES = {
    "Simply Supported":   {"left": "pin",    "right": "roller", "F_at": "mid"},
    "Cantilever":         {"left": "fixed",  "right": "free",   "F_at": "end"},
    "Fixed-Fixed":        {"left": "fixed",  "right": "fixed",  "F_at": "mid"},
    "Propped Cantilever": {"left": "fixed",  "right": "pin",    "F_at": "mid"},
    "Elastic Support":    {"left": "spring", "right": "spring", "F_at": "mid"},
    "Overhang":           {"left": "pin",    "right": "free",   "F_at": "end", "inner": "roller"},
    "Custom":             {"left": "free",   "right": "free",   "F_at": "mid"},
}

def get_preset_nodes(beam_type, nel, L, F_val=-1000.0, k_spring=50.0):
    """Return list of node dicts with preset BCs."""
    le = L / nel
    nN = nel + 1
    mid = nel // 2
    bt = BEAM_TYPES[beam_type]

    nodes = []
    for i in range(nN):
        nodes.append({"x": round(i * le, 4), "disp": None, "rot": None,
                      "F": 0.0, "M": 0.0, "k": 0.0, "kr": 0.0})

    def apply_bc(idx, bc, k_val=k_spring):
        if bc == "pin":
            nodes[idx]["disp"] = 0.0
        elif bc == "roller":
            nodes[idx]["disp"] = 0.0
        elif bc == "fixed":
            nodes[idx]["disp"] = 0.0
            nodes[idx]["rot"]  = 0.0
        elif bc == "spring":
            nodes[idx]["k"] = k_val

    apply_bc(0,       bt["left"])
    apply_bc(nN - 1,  bt["right"])
    if "inner" in bt:
        sp = max(1, min(int(nN * 0.6), nN - 2))
        apply_bc(sp, bt["inner"])

    # load
    fat = bt.get("F_at", "mid")
    load_idx = nN - 1 if fat == "end" else mid
    nodes[load_idx]["F"] = F_val
    return nodes

# ─────────────────────────────────────────
# FEM SOLVER
# ─────────────────────────────────────────
def assemble_and_solve(nodes, nel, E, I_val):
    nN  = nel + 1
    ndof = 2 * nN
    L_list = [nodes[e+1]["x"] - nodes[e]["x"] for e in range(nel)]

    K = np.zeros((ndof, ndof))
    for e in range(nel):
        le = L_list[e]
        EI = E * I_val
        c  = EI / le**3
        ke = c * np.array([
            [12,     6*le,  -12,    6*le ],
            [6*le,   4*le**2, -6*le, 2*le**2],
            [-12,   -6*le,   12,   -6*le ],
            [6*le,   2*le**2, -6*le, 4*le**2],
        ])
        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += ke[i, j]

    # springs
    for i, nd in enumerate(nodes):
        if nd["k"]  > 0: K[2*i,   2*i  ] += nd["k"]
        if nd["kr"] > 0: K[2*i+1, 2*i+1] += nd["kr"]

    # load vector
    F = np.zeros(ndof)
    for i, nd in enumerate(nodes):
        F[2*i]   = nd["F"]
        F[2*i+1] = nd["M"]

    # BCs — penalty / elimination
    prescribed = []
    for i, nd in enumerate(nodes):
        if nd["disp"] is not None: prescribed.append((2*i,   nd["disp"]))
        if nd["rot"]  is not None: prescribed.append((2*i+1, nd["rot"]))

    if not prescribed:
        raise ValueError("No boundary conditions defined. Set at least one prescribed displacement or rotation.")

    Km = K.copy(); Fm = F.copy()
    for dof, val in prescribed:
        Fm -= Km[:, dof] * val
        Km[dof, :] = 0
        Km[:, dof] = 0
        Km[dof, dof] = 1
        Fm[dof] = val

    u = np.linalg.solve(Km, Fm)
    return K, F, u

# ─────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────
def hermite_interp(e, xi, u_e, le):
    """Transverse displacement at local xi in [0,1]."""
    w1, th1, w2, th2 = u_e
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = le * xi * (1 - xi)**2
    N3 = 3*xi**2 - 2*xi**3
    N4 = le * xi**2 * (xi - 1)
    return N1*w1 + N2*th1 + N3*w2 + N4*th2

def hermite_curvature(xi, u_e, le):
    """κ = d²w/dx² (curvature) from Hermite 2nd derivatives."""
    w1, th1, w2, th2 = u_e
    d2N1 = (-6 + 12*xi) / le**2
    d2N2 = (-4 + 6*xi)  / le
    d2N3 = ( 6 - 12*xi) / le**2
    d2N4 = (-2 + 6*xi)  / le
    return d2N1*w1 + d2N2*th1 + d2N3*w2 + d2N4*th2

def compute_results(u, nodes, nel, E, I_val, W_val, n_pts=12):
    xs, ws, Ms, ss = [], [], [], []
    for e in range(nel):
        le = nodes[e+1]["x"] - nodes[e]["x"]
        u_e = [u[2*e], u[2*e+1], u[2*e+2], u[2*e+3]]
        for p in range(n_pts + (1 if e == nel-1 else 0)):
            xi = p / n_pts
            x_abs = nodes[e]["x"] + xi * le
            w   = hermite_interp(e, xi, u_e, le)
            kap = hermite_curvature(xi, u_e, le)
            M   = E * I_val * kap
            sig = M / W_val
            xs.append(x_abs); ws.append(w); Ms.append(M); ss.append(sig)
    return np.array(xs), np.array(ws), np.array(Ms), np.array(ss)

# ─────────────────────────────────────────
# MATPLOTLIB FIGURES
# ─────────────────────────────────────────
BLUE   = "#3d77ca"
LBLUE  = "#91b3ed"
RED    = "#cc2222"
ORANGE = "#e08030"
GREY   = "#848484"

def fig_beam_sketch(beam_type, F_val=-1000):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.set_xlim(0, 10); ax.set_ylim(-1.2, 2.5)
    ax.axis("off")
    x0, x1, ymid = 1.0, 9.0, 0.5
    bh = 0.35
    ax.add_patch(mpatches.FancyBboxPatch((x0, ymid - bh/2), x1-x0, bh,
                 boxstyle="square,pad=0", fc=BLUE, ec="none", zorder=3))

    bt = BEAM_TYPES[beam_type]

    def pin_support(x):
        tri = plt.Polygon([[x, ymid-bh/2],[x-0.5, ymid-bh/2-0.7],[x+0.5, ymid-bh/2-0.7]],
                          fc=BLUE, ec=BLUE, zorder=2)
        ax.add_patch(tri)
        ax.plot([x-0.6, x+0.6], [ymid-bh/2-0.75]*2, color=BLUE, lw=2)

    def roller_support(x):
        tri = plt.Polygon([[x, ymid-bh/2],[x-0.5, ymid-bh/2-0.6],[x+0.5, ymid-bh/2-0.6]],
                          fc=LBLUE, ec=BLUE, lw=1.2, zorder=2)
        ax.add_patch(tri)
        ax.add_patch(plt.Circle((x-0.3, ymid-bh/2-0.75), 0.12, fc="none", ec=BLUE, lw=1.2))
        ax.add_patch(plt.Circle((x+0.3, ymid-bh/2-0.75), 0.12, fc="none", ec=BLUE, lw=1.2))

    def fixed_support(x, side="l"):
        ax.plot([x, x], [ymid-bh/2-0.5, ymid+bh/2+0.5], color=BLUE, lw=4, zorder=4)
        d = -0.4 if side == "l" else 0.4
        for dy in np.linspace(-0.4, 0.4, 5):
            ax.plot([x, x+d], [ymid+dy, ymid+dy+0.2], color=LBLUE, lw=1)

    def spring_support(x):
        ys = np.linspace(ymid-bh/2, ymid-bh/2-0.9, 12)
        xs_ = [x + (0.3 if i%2==0 else -0.3) for i in range(len(ys))]
        ax.plot(xs_, ys, color=ORANGE, lw=2)
        ax.plot([x-0.4, x+0.4], [ymid-bh/2-0.92]*2, color=ORANGE, lw=1.5)

    def load_arrow(x, label="F"):
        tip_y  = ymid + bh/2
        tail_y = tip_y + 1.0
        ax.annotate("", xy=(x, tip_y), xytext=(x, tail_y),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2,
                                   mutation_scale=18))
        ax.text(x, tail_y + 0.1, label, ha="center", va="bottom",
                color=RED, fontsize=9, fontweight="bold")

    for bc, fn_l, fn_r in [
        ("pin",    pin_support,   None),
        ("roller", roller_support, None),
        ("fixed",  fixed_support,  None),
        ("spring", spring_support, None),
    ]:
        if bt["left"]  == bc: fn_l(x0)
        if bt["right"] == bc: (fn_r or fn_l)(x1)

    if bt["left"]  == "fixed": fixed_support(x0, "l")
    if bt["right"] == "fixed": fixed_support(x1, "r")

    load_x = x1 if bt.get("F_at") == "end" else (x0+x1)/2
    load_arrow(load_x, f"F = {int(F_val)} N")

    if beam_type == "Overhang":
        sp_x = x0 + (x1-x0)*0.6
        roller_support(sp_x)
        ax.text((sp_x+x1)/2, ymid-bh/2-0.2, "overhang",
                ha="center", va="top", fontsize=7, color=GREY)

    ax.set_title(beam_type, fontsize=10, color=BLUE, fontweight="bold", pad=3)
    fig.tight_layout(pad=0.3)
    return fig


def fig_model(nodes, nel):
    nN = nel + 1
    L  = nodes[-1]["x"]
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(-L*0.08, L*1.08)
    ax.set_ylim(-L*0.18, L*0.22)
    ax.axis("off")
    bh = L * 0.04

    # beam segments
    for e in range(nel):
        x0e = nodes[e]["x"]; x1e = nodes[e+1]["x"]
        col = BLUE if e%2==0 else "#2d67ba"
        ax.add_patch(mpatches.FancyBboxPatch((x0e, -bh/2), x1e-x0e, bh,
                     boxstyle="square,pad=0", fc=col, ec="none", zorder=2))
        ax.text((x0e+x1e)/2, 0, f"e{e+1}", ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=3)

    for nd in nodes:
        x  = nd["x"]
        bx = x

        # supports
        if nd["disp"] == 0 and nd["rot"] == 0:
            ax.plot([x, x], [-bh/2-L*0.06, bh/2+L*0.06], color=BLUE, lw=3, zorder=4)
            for dy in np.linspace(-bh*1.2, bh*1.2, 4):
                ax.plot([x, x - L*0.03], [dy, dy + L*0.02], color=LBLUE, lw=0.8)
        elif nd["disp"] == 0:
            tri_h = L*0.07
            tri = plt.Polygon([[x, -bh/2],[x-L*0.04, -bh/2-tri_h],[x+L*0.04, -bh/2-tri_h]],
                              fc=BLUE, ec=BLUE, zorder=2)
            ax.add_patch(tri)
            ax.plot([x-L*0.05, x+L*0.05], [-bh/2-tri_h-L*0.008]*2, color=BLUE, lw=1.5)
        elif nd["k"] > 0:
            ys = np.linspace(-bh/2, -bh/2 - L*0.12, 10)
            xs_ = [x + (L*0.025 if i%2==0 else -L*0.025) for i in range(10)]
            ax.plot(xs_, ys, color=ORANGE, lw=1.8, zorder=3)
            ax.plot([x-L*0.035, x+L*0.035], [-bh/2-L*0.123]*2, color=ORANGE, lw=1.5)
            ax.text(x, -bh/2 - L*0.145, f"k={nd['k']:.0f}", ha="center",
                    va="top", fontsize=6, color=ORANGE)
        else:
            ax.add_patch(plt.Circle((x, 0), bh*0.35, fc="white", ec=BLUE, lw=1.2, zorder=4))

        # force arrow — tip at top of beam, pointing DOWN
        if nd["F"] != 0:
            tip_y  =  bh/2
            tail_y =  bh/2 + L*0.14
            ax.annotate("", xy=(x, tip_y), xytext=(x, tail_y),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, mutation_scale=14))
            ax.text(x, tail_y + L*0.015, f"{nd['F']:.0f}N",
                    ha="center", va="bottom", color=RED, fontsize=7, fontweight="bold")

        # moment arc
        if nd["M"] != 0:
            ang = np.linspace(0, np.pi, 30)
            sign = 1 if nd["M"] > 0 else -1
            ax.plot(x + L*0.04*np.cos(ang), L*0.04*np.sin(ang)*sign,
                    color="#8040c0", lw=1.8)

        # node label
        ax.text(x, -bh/2 - L*0.09, str(nodes.index(nd)+1),
                ha="center", va="top", fontsize=7, color=BLUE, fontweight="bold")
        ax.text(x, -bh/2 - L*0.14, f"{x:.0f}mm",
                ha="center", va="top", fontsize=6, color=GREY)

    ax.text(L*1.02, 0, "x →", ha="left", va="center", fontsize=8, color=GREY)
    ax.set_title(f"Model: {nel} elements · {nN} nodes", fontsize=9, color=BLUE, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def fig_deformed(nodes, nel, u, xs_fine, ws_fine):
    nN = nel + 1
    L  = nodes[-1]["x"]
    max_w = np.max(np.abs(ws_fine))
    if max_w < 1e-12: max_w = 1.0
    scale = (L * 0.18) / max_w

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(-L*0.05, L*1.05)
    ax.axhline(0, color=LBLUE, lw=1, ls="--", label="Undeformed")
    ax.plot(xs_fine, ws_fine * scale, color=RED, lw=2.2, label="Deformed (scaled)")
    # node markers
    for i in range(nN):
        xi = nodes[i]["x"]
        wi = u[2*i] * scale
        ax.plot(xi, wi, "o", color=BLUE, ms=6, zorder=5)
    ax.set_xlabel("x (mm)", fontsize=9, color=GREY)
    ax.set_ylabel("w (scaled)", fontsize=9, color=GREY)
    ax.set_title(f"Deformed Shape  [max |w| = {max_w:.4g} mm]", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_moment(xs, Ms):
    L  = xs[-1]
    max_M = np.max(np.abs(Ms))
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.fill_between(xs, 0, Ms, where=Ms >= 0, alpha=0.25, color=BLUE, label="M > 0")
    ax.fill_between(xs, 0, Ms, where=Ms <  0, alpha=0.25, color=RED,  label="M < 0")
    ax.plot(xs, Ms, color=BLUE, lw=2.0)
    ax.axhline(0, color=LBLUE, lw=1, ls="--")
    ax.set_xlabel("x (mm)", fontsize=9, color=GREY)
    ax.set_ylabel("M (Nmm)", fontsize=9, color=GREY)
    ax.set_title(f"Bending Moment Diagram  [max |M| = {max_M:.4g} Nmm]", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_stress(xs, ss):
    max_s = np.max(np.abs(ss))
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.fill_between(xs, 0, ss, where=ss >= 0, alpha=0.22, color=BLUE, label="σ > 0 (tension)")
    ax.fill_between(xs, 0, ss, where=ss <  0, alpha=0.22, color=RED,  label="σ < 0 (compression)")
    ax.plot(xs, ss, color=BLUE, lw=2.0)
    ax.axhline(0, color=LBLUE, lw=1, ls="--")
    ax.set_xlabel("x (mm)", fontsize=9, color=GREY)
    ax.set_ylabel("σ (MPa)", fontsize=9, color=GREY)
    ax.set_title(f"Bending Stress  σ = M/W  [max |σ| = {max_s:.4g} MPa]", fontsize=10, color=BLUE)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")
    st.markdown('<div class="sidebar-note">Configure beam, material and section, then set loads in the main panel.</div>', unsafe_allow_html=True)

    # ── Beam Type ──
    beam_type = st.selectbox("Beam Type", list(BEAM_TYPES.keys()), index=0)

    st.markdown("---")
    st.markdown("### Material")
    E = st.number_input("Young's Modulus E (N/mm²)", value=210000, step=1000, min_value=1)

    st.markdown("### Geometry")
    L = st.number_input("Total Length L (mm)", value=1000, step=100, min_value=10)
    nel = st.selectbox("Number of Elements", [2, 3, 4, 5, 6, 8, 10], index=1)

    st.markdown("### Cross-Section")
    profile = st.selectbox("Profile Type", ["Rectangle", "Circle", "I-Beam", "Hollow Rectangle"])
    b = st.number_input("Width b (mm)", value=40.0, step=1.0, min_value=1.0)
    h = st.number_input("Height h (mm)", value=80.0, step=1.0, min_value=1.0)
    t_extra = None
    if profile in ["I-Beam", "Hollow Rectangle"]:
        t_extra = st.number_input("Wall/Flange thickness t (mm)", value=5.0, step=0.5, min_value=0.5)

    I_auto, W_auto, y_max = calc_section(profile, b, h, t_extra)
    use_auto = st.checkbox("Use auto-calculated I & W", value=True)
    if use_auto:
        I_val = I_auto; W_val = W_auto
        st.info(f"I = {I_auto:,.0f} mm⁴   W = {W_auto:,.0f} mm³")
    else:
        I_val = st.number_input("I (mm⁴)", value=float(I_auto), step=1000.0, min_value=1.0)
        W_val = st.number_input("W (mm³)", value=float(W_auto), step=100.0,  min_value=1.0)

# Sidebar Support Section
st.sidebar.markdown("---")
st.sidebar.subheader("Support the Virtual Lab")

st.sidebar.write("""
If these simulations assist in your teaching or learning, your support helps maintain the server and develop new educational modules.
""")

with st.sidebar:
    import streamlit.components.v1 as components
    
# 1. PayPal Option (International Support)
# Using a direct URL link ensures compatibility across all browsers and devices
paypal_url = "https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=DJXQVFP3FHQDN"

st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 15px;">
    <p style="font-size: 13px; margin-bottom: 8px;"><b>International Support (USD)</b></p>
    <a href="{paypal_url}" target="_blank" style="text-decoration: none;">
        <img src="https://www.paypalobjects.com/webstatic/en_US/i/buttons/checkout-logo-medium.png" 
             alt="Donate with PayPal" style="width: 140px; border: 0;">
    </a>
</div>
""", unsafe_allow_html=True)

# 2. Local Support (Myanmar KPay)
st.sidebar.markdown("""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #d1d5db;">
    <p style="font-size: 13px; margin-bottom: 2px;"><b>Local Support (KPay)</b></p>
    <p style="font-size: 14px; font-weight: bold; color: #1e3a8a; margin: 0;">09793490979</p>
    <p style="font-size: 11px; color: #6b7280; margin: 0;">Theingi Nwe</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Section
st.sidebar.markdown("---")
st.sidebar.subheader("Developed by: Theingi Nwe")

# Highlighting the link with an icon and a border for a "button" feel
st.sidebar.info("🌐 [Finite Element Institute (Official Blog)](https://fesimulationsbytgn.blogspot.com)")

        
# ─────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────
st.title("🏗️ 1D Beam FEM Simulator")
st.caption("Euler-Bernoulli beam · Hermitian cubic shape functions · Gaussian elimination solver")

tab_setup, tab_theory, tab_matrix, tab_results = st.tabs(
    ["⚙️ Setup & Solve", "📚 Theory", "🔢 Matrix", "📊 Results"])

# ─────────────────────────────────────────────────────
# TAB 1: SETUP
# ─────────────────────────────────────────────────────
with tab_setup:
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Beam Type Preview")
        F_default = -1000.0
        fig_bt = fig_beam_sketch(beam_type, F_default)
        st.pyplot(fig_bt, use_container_width=True)
        plt.close(fig_bt)

    with col2:
        st.subheader("Section Properties")
        st.metric("Moment of Inertia I", f"{I_val:,.0f} mm⁴")
        st.metric("Section Modulus W",   f"{W_val:,.0f} mm³")
        st.metric("EI",                  f"{E*I_val:,.2e} Nmm²")

    st.markdown("---")

    # ── Node Definition Table ──
    st.subheader("Node Definition")
    st.markdown("""
    | Symbol | Meaning |
    |--------|---------|
    | **Disp [mm]** | Prescribed transverse displacement — leave blank = free |
    | **Rot [rad]** | Prescribed rotation — leave blank = free |
    | **F [N]** | Nodal force (downward = **negative**) |
    | **M [Nmm]** | Nodal bending moment |
    | **k [N/mm]** | Translational spring stiffness |
    | **kr [Nmm/rad]** | Rotational spring stiffness |
    """)

    nN = nel + 1
    le = L / nel

    # session state for nodes
    key = f"{beam_type}_{nel}_{L}"
    if "node_key" not in st.session_state or st.session_state["node_key"] != key:
        st.session_state["node_key"]   = key
        st.session_state["preset_nodes"] = get_preset_nodes(beam_type, nel, L, F_val=-1000.0)

    preset_nodes = st.session_state["preset_nodes"]

    # build editable dataframe
    def nodes_to_df(nds):
        rows = []
        for nd in nds:
            rows.append({
                "Node": nds.index(nd) + 1,
                "x [mm]": nd["x"],
                "Disp [mm]": nd["disp"] if nd["disp"] is not None else "",
                "Rot [rad]": nd["rot"]  if nd["rot"]  is not None else "",
                "F [N]":  nd["F"],
                "M [Nmm]": nd["M"],
                "k [N/mm]": nd["k"],
                "kr [Nmm/rad]": nd["kr"],
            })
        return pd.DataFrame(rows)

    df_default = nodes_to_df(preset_nodes)

    edited_df = st.data_editor(
        df_default,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Node":    st.column_config.NumberColumn("Node", disabled=True, width="small"),
            "x [mm]": st.column_config.NumberColumn("x [mm]", disabled=True, width="small"),
            "Disp [mm]":    st.column_config.TextColumn("Disp [mm]",   help="blank = free DOF"),
            "Rot [rad]":    st.column_config.TextColumn("Rot [rad]",   help="blank = free DOF"),
            "F [N]":         st.column_config.NumberColumn("F [N]",     step=0.1),
            "M [Nmm]":       st.column_config.NumberColumn("M [Nmm]",   step=0.1),
            "k [N/mm]":      st.column_config.NumberColumn("k [N/mm]",  step=0.1, min_value=0.0),
            "kr [Nmm/rad]":  st.column_config.NumberColumn("kr [Nmm/rad]", step=0.1, min_value=0.0),
        },
        key="node_editor",
    )

    # parse edited dataframe back to nodes
    def df_to_nodes(df):
        nds = []
        for _, row in df.iterrows():
            def parse_bc(v):
                if v == "" or v is None or (isinstance(v, float) and np.isnan(v)): return None
                try: return float(v)
                except: return None
            nds.append({
                "x":    float(row["x [mm]"]),
                "disp": parse_bc(row["Disp [mm]"]),
                "rot":  parse_bc(row["Rot [rad]"]),
                "F":    float(row["F [N]"]),
                "M":    float(row["M [Nmm]"]),
                "k":    float(row["k [N/mm]"]),
                "kr":   float(row["kr [Nmm/rad]"]),
            })
        return nds

    current_nodes = df_to_nodes(edited_df)

    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        solve_clicked = st.button("▶ Solve FEM", type="primary", use_container_width=True)
    with bcol2:
        reset_clicked = st.button("↺ Reset to Preset", use_container_width=False)

    if reset_clicked:
        st.session_state["node_key"] = ""
        st.rerun()

    # ── Model Preview ──
    st.markdown("---")
    st.subheader("Model Preview")
    fig_m = fig_model(current_nodes, nel)
    st.pyplot(fig_m, use_container_width=True)
    plt.close(fig_m)

    # ── SOLVE ──
    if solve_clicked:
        try:
            K, Fvec, u = assemble_and_solve(current_nodes, nel, E, I_val)
            xs_f, ws_f, Ms_f, ss_f = compute_results(u, current_nodes, nel, E, I_val, W_val)

            st.session_state["solved"]  = True
            st.session_state["K"]       = K
            st.session_state["Fvec"]    = Fvec
            st.session_state["uvec"]    = u
            st.session_state["nodes"]   = current_nodes
            st.session_state["nel"]     = nel
            st.session_state["xs_f"]    = xs_f
            st.session_state["ws_f"]    = ws_f
            st.session_state["Ms_f"]    = Ms_f
            st.session_state["ss_f"]    = ss_f

            max_w = float(np.max(np.abs(u[::2])))
            max_M = float(np.max(np.abs(Ms_f)))
            max_s = float(np.max(np.abs(ss_f)))

            st.success(f"✅ Solved! Max |w| = {max_w:.4g} mm  |  Max |M| = {max_M:.4g} Nmm  |  Max |σ| = {max_s:.4g} MPa  →  Switch to **Results** tab.")
        except Exception as ex:
            st.error(f"❌ Solver error: {ex}")

# ─────────────────────────────────────────────────────
# TAB 2: THEORY
# ─────────────────────────────────────────────────────
with tab_theory:
    st.subheader("Euler-Bernoulli Beam Theory")

    st.markdown("#### Governing PDE")
    st.latex(r"EI \frac{d^4 w}{dx^4} = q(x)")
    st.caption("E = Young's modulus · I = 2nd moment of area · w(x) = transverse deflection · q = distributed load")

    st.markdown("#### Hermitian Cubic Shape Functions  (ξ = x/Lₑ ∈ [0,1])")
    col_a, col_b = st.columns(2)
    with col_a:
        st.latex(r"N_1(\xi) = 1 - 3\xi^2 + 2\xi^3")
        st.latex(r"N_2(\xi) = L_e\,\xi\,(1-\xi)^2")
    with col_b:
        st.latex(r"N_3(\xi) = 3\xi^2 - 2\xi^3")
        st.latex(r"N_4(\xi) = L_e\,\xi^2\,(\xi - 1)")
    st.caption("DOF vector per element: [w₁, θ₁, w₂, θ₂] — transverse displacement and rotation at each node")

    st.markdown("#### Element Stiffness Matrix")
    st.latex(r"""
    \mathbf{k}_e = \frac{EI}{L_e^3}
    \begin{bmatrix}
    12    &  6L_e  & -12   &  6L_e \\
    6L_e  &  4L_e^2& -6L_e &  2L_e^2 \\
    -12   & -6L_e  &  12   & -6L_e \\
    6L_e  &  2L_e^2& -6L_e &  4L_e^2
    \end{bmatrix}
    """)

    st.markdown("#### Global System")
    st.latex(r"\mathbf{K}\,\mathbf{u} = \mathbf{F}")
    st.caption("Boundary conditions applied by row/column elimination (exact method). Solved by numpy.linalg.solve (LU decomposition).")

    st.markdown("#### Curvature and Bending Moment")
    st.latex(r"\kappa(x) = \frac{d^2 w}{dx^2} = \frac{1}{L_e^2}\,\mathbf{B}(\xi)\cdot\mathbf{u}_e")
    st.latex(r"M(x) = EI\,\kappa(x)")

    st.markdown("#### Bending Stress")
    st.latex(r"\sigma(x) = \frac{M(x)}{W} \quad \text{where} \quad W = \frac{I}{y_{max}}")
    st.caption("Maximum stress at extreme fibres y = ±h/2. In Euler-Bernoulli theory shear stress is neglected → von Mises = |σ_bending|")



# ─────────────────────────────────────────────────────
# TAB 3: MATRIX
# ─────────────────────────────────────────────────────
with tab_matrix:
    if st.session_state.get("solved"):
        K   = st.session_state["K"]
        Fv  = st.session_state["Fvec"]
        uv  = st.session_state["uvec"]
        nds = st.session_state["nodes"]
        n   = len(uv)

        st.subheader(f"Global Stiffness Matrix K  ({n}×{n})")
        dof_labels = []
        for i in range(n//2):
            dof_labels += [f"w{i+1}", f"θ{i+1}"]
        df_K = pd.DataFrame(K, index=dof_labels, columns=dof_labels)
        st.dataframe(df_K.style.format("{:.3e}").highlight_between(
            left=-1e-9, right=1e-9, props="background-color:#f0f8ff;"),
            use_container_width=True, height=min(40+n*34, 500))

        st.subheader("Load Vector F  &  Displacement Vector u")
        df_Fu = pd.DataFrame({
            "DOF":     dof_labels,
            "Meaning": ["disp [N]" if i%2==0 else "rot [Nmm]" for i in range(n)],
            "F value": [f"{Fv[i]:.4e}" for i in range(n)],
            "u value": [f"{uv[i]:.6e}" for i in range(n)],
        })
        st.dataframe(df_Fu, use_container_width=True)

        # Download
        st.subheader("Downloads")
        c1, c2 = st.columns(2)
        with c1:
            csv_K = df_K.to_csv().encode()
            st.download_button("⬇ Download K matrix (CSV)", csv_K, "K_matrix.csv", "text/csv")
        with c2:
            csv_Fu = df_Fu.to_csv(index=False).encode()
            st.download_button("⬇ Download F & u vectors (CSV)", csv_Fu, "F_u_vectors.csv", "text/csv")
    else:
        st.info("Solve FEM first (Setup & Solve tab).")

# ─────────────────────────────────────────────────────
# TAB 4: RESULTS
# ─────────────────────────────────────────────────────
with tab_results:
    if st.session_state.get("solved"):
        uv   = st.session_state["uvec"]
        nds  = st.session_state["nodes"]
        nel_ = st.session_state["nel"]
        xs_f = st.session_state["xs_f"]
        ws_f = st.session_state["ws_f"]
        Ms_f = st.session_state["Ms_f"]
        ss_f = st.session_state["ss_f"]

        nN_ = nel_ + 1
        wArr  = uv[::2]
        thArr = uv[1::2]
        max_w = float(np.max(np.abs(wArr)))
        max_M = float(np.max(np.abs(Ms_f)))
        max_s = float(np.max(np.abs(ss_f)))

        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Max |Displacement|", f"{max_w:.4g} mm")
        m2.metric("Max |Bending Moment|", f"{max_M:.4g} Nmm")
        m3.metric("Max |Bending Stress|", f"{max_s:.4g} MPa")

        st.markdown("---")

        # ── Deformed shape ──
        st.subheader("Beam Displacement — Deformed Shape")
        fig_d = fig_deformed(nds, nel_, uv, xs_f, ws_f)
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        # ── Bending Moment ──
        st.subheader("Bending Moment Diagram")
        fig_bm = fig_moment(xs_f, Ms_f)
        st.pyplot(fig_bm, use_container_width=True)
        plt.close(fig_bm)

        # ── Stress ──
        st.subheader("Bending Stress Diagram  (σ = M/W)")
        fig_s = fig_stress(xs_f, ss_f)
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)

        st.markdown("---")
        # ── Result Table ──
        st.subheader("Result Table — Node Values")
        rows = []
        for i in range(nN_):
            rows.append({
                "Node": i+1,
                "x [mm]": round(nds[i]["x"], 3),
                "w [mm]":  float(f"{wArr[i]:.6e}"),
                "θ [rad]": float(f"{thArr[i]:.6e}"),
                "F [N]":   nds[i]["F"],
                "BC disp": "" if nds[i]["disp"] is None else nds[i]["disp"],
                "BC rot":  "" if nds[i]["rot"]  is None else nds[i]["rot"],
            })
        df_res = pd.DataFrame(rows)
        st.dataframe(df_res.style.format({
            "w [mm]":  "{:.4e}", "θ [rad]": "{:.4e}"}),
            use_container_width=True)

        # ── Fine result table ──
        st.subheader("Result Table — 100 Points Along Beam")
        n100 = 100
        xs100  = np.linspace(xs_f[0], xs_f[-1], n100)
        ws100  = np.interp(xs100, xs_f, ws_f)
        Ms100  = np.interp(xs100, xs_f, Ms_f)
        ss100  = np.interp(xs100, xs_f, ss_f)
        df_100 = pd.DataFrame({"x [mm]":xs100,"w [mm]":ws100,"M [Nmm]":Ms100,"σ [MPa]":ss100})
        st.dataframe(df_100.style.format("{:.4e}"), use_container_width=True, height=300)

        # Downloads
        st.subheader("Downloads")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            buf_d = fig_to_bytes(fig_deformed(nds, nel_, uv, xs_f, ws_f))
            st.download_button("⬇ Deformed Shape (PNG)", buf_d, "deformed_shape.png", "image/png")
        with dc2:
            buf_m = fig_to_bytes(fig_moment(xs_f, Ms_f))
            st.download_button("⬇ Moment Diagram (PNG)", buf_m, "moment_diagram.png", "image/png")
        with dc3:
            buf_s = fig_to_bytes(fig_stress(xs_f, ss_f))
            st.download_button("⬇ Stress Diagram (PNG)", buf_s, "stress_diagram.png", "image/png")
        csv_100 = df_100.to_csv(index=False).encode()
        st.download_button("⬇ Full Result Table (CSV)", csv_100, "results_100pts.csv", "text/csv")
    else:
        st.info("Solve FEM first (Setup & Solve tab).")

# ── init session state ──
if "solved" not in st.session_state:
    st.session_state["solved"] = False
