import os
import io
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Configuration
# =========================
OUTPUT_DIR_DEFAULT = "output"

# Expected column names from the fetcher script
GENESIS_COL = "CyberKongz Genesis"
BABIES_COL  = "CyberKongz Babies"
VX_ETH_COL  = "CyberKongz VX"
VX_RON_COL  = "CyberKongz VX (Ronin)"
GK_ETH_COL  = "CyberKongz Genkai"
GK_RON_COL  = "CyberKongz Genkai (Ronin)"
VX_ANY_COL  = "CyberKongz VX (Any)"
GK_ANY_COL  = "CyberKongz Genkai (Any)"

# =========================
# Page config
# =========================
st.set_page_config(page_title="CyberKongz — Holders & Overlap", page_icon="🦍", layout="wide")

# Small CSS for the highlight card
st.markdown(
    """
    <style>
    .hi-box {
      background: linear-gradient(90deg, #2a9d8f22, #2a9d8f33);
      border: 1px solid #2a9d8f66;
      padding: 16px 20px;
      border-radius: 14px;
    }
    .hi-title {
      margin: 0 0 4px 0;
      font-size: 1rem;
      opacity: .85;
    }
    .hi-value {
      margin: 0;
      font-size: 3rem !important;   /* force bigger */
      font-weight: 900 !important;  /* force bolder */
      color: #ffffff !important;    /* force white */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Data helpers
# =========================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_all(output_dir: str):
    holdings = load_csv(os.path.join(output_dir, "per_wallet_holdings.csv"))
    overlap  = load_csv(os.path.join(output_dir, "overlap_summary.csv"))
    members  = load_csv(os.path.join(output_dir, "combination_membership.csv"))
    return holdings, overlap, members

def ensure_wallet_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "wallet" in df.columns:
        return df
    if df.index.name == "wallet":
        return df.reset_index()
    return df

def file_age_str(path: str) -> str:
    if not os.path.exists(path):
        return "—"
    t = dt.datetime.fromtimestamp(os.path.getmtime(path))
    return t.strftime("%Y-%m-%d %H:%M")

def add_missing_any_columns(mat: pd.DataFrame) -> pd.DataFrame:
    """If '(Any)' columns are missing, synthesize them from ETH+Ronin; else leave as-is."""
    mat = mat.copy()
    if VX_ANY_COL not in mat.columns:
        vx_any = mat.get(VX_ETH_COL, 0).fillna(0).astype(int) + mat.get(VX_RON_COL, 0).fillna(0).astype(int)
        mat[VX_ANY_COL] = vx_any
    if GK_ANY_COL not in mat.columns:
        gk_any = mat.get(GK_ETH_COL, 0).fillna(0).astype(int) + mat.get(GK_RON_COL, 0).fillna(0).astype(int)
        mat[GK_ANY_COL] = gk_any
    return mat

def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return "0"

def download_df_button(df: pd.DataFrame, label: str, filename: str):
    if df.empty:
        st.button(label, disabled=True, width="stretch")
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label,
        buf.getvalue().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )

# =========================
# Header (no sidebar, hardcoded output path)
# =========================
st.title("🦍 CyberKongz — Holders & Overlap")

out_dir = OUTPUT_DIR_DEFAULT  # hardcoded path

pf = os.path.join(out_dir, "per_wallet_holdings.csv")
st.caption(f"Data last updated: {file_age_str(pf)}")

# Remarks box
st.markdown(
    """
    <div style='background-color:#fff3cd; color:#856404; padding:10px; border-radius:6px;'>
        <b>Remarks:</b> Polygon VX not included
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# Load & normalize data
# =========================
holdings, overlap, members = load_all(out_dir)

holdings = ensure_wallet_col(holdings)
members  = ensure_wallet_col(members)

if holdings.empty or "wallet" not in holdings.columns:
    st.warning("No data found. Make sure `per_wallet_holdings.csv` exists in the selected output folder.")
    st.stop()

# Ensure (Any) columns are present
holdings = add_missing_any_columns(holdings)

# =========================
# Top KPI (wallet-holder counts)
# =========================
k1, k2, k3, k4 = st.columns(4)
distinct_wallets = holdings["wallet"].nunique()
any_genesis = (holdings.get(GENESIS_COL, 0) > 0).sum()
any_babies  = (holdings.get(BABIES_COL, 0)  > 0).sum()
any_vx      = (holdings.get(VX_ANY_COL, 0)  > 0).sum()
any_genkai  = (holdings.get(GK_ANY_COL, 0)  > 0).sum()

with k1: st.metric("Distinct wallets", fmt_int(distinct_wallets))
with k2: st.metric("Any Genesis", fmt_int(any_genesis))
with k3: st.metric("Any Babies", fmt_int(any_babies))
with k4: st.metric("Any VX / Genkai", f"{fmt_int(any_vx)} / {fmt_int(any_genkai)}")

# =========================
# Tabs (Threshold first, then Overview, Downloads)
# =========================
tab_thresholds, tab_overview, tab_downloads = st.tabs(
    ["Threshold query", "Overview", "Downloads"]
)

# -------- Threshold query --------
with tab_thresholds:
    st.subheader("Holding treshholds")

    def col_safe(name: str):
        return (holdings[name] if name in holdings.columns else 0)

    # Inputs: Min + Max (0 = no upper limit)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_genesis = st.number_input("Min Genesis (ETH)", min_value=0, value=0, step=1)
        max_genesis = st.number_input("Max Genesis (ETH) — 0 = ∞", min_value=0, value=0, step=1)
    with c2:
        min_babies  = st.number_input("Min Babies (ETH)",  min_value=0, value=0, step=1)
        max_babies  = st.number_input("Max Babies (ETH) — 0 = ∞",  min_value=0, value=0, step=1)
    with c3:
        min_vx_any  = st.number_input("Min VX (Any chain)", min_value=0, value=0, step=1)
        max_vx_any  = st.number_input("Max VX (Any) — 0 = ∞", min_value=0, value=0, step=1)
    with c4:
        min_gk_any  = st.number_input("Min Genkai (Any chain)", min_value=0, value=0, step=1)
        max_gk_any  = st.number_input("Max Genkai (Any) — 0 = ∞", min_value=0, value=0, step=1)

    # Series
    s_gen = col_safe(GENESIS_COL).fillna(0).astype(int)
    s_bab = col_safe(BABIES_COL).fillna(0).astype(int)
    s_vx  = col_safe(VX_ANY_COL).fillna(0).astype(int)
    s_gk  = col_safe(GK_ANY_COL).fillna(0).astype(int)

    def within_bounds(s: pd.Series, mn: int, mx: int) -> pd.Series:
        upper_ok = (s <= mx) if mx > 0 else True
        return (s >= mn) & upper_ok

    # Per-asset masks respecting min & max
    m_gen = within_bounds(s_gen, min_genesis, max_genesis)
    m_bab = within_bounds(s_bab, min_babies,  max_babies)
    m_vx  = within_bounds(s_vx,  min_vx_any,  max_vx_any)
    m_gk  = within_bounds(s_gk,  min_gk_any,  max_gk_any)

    # Counts per-asset (non-exclusive)
    c_gen = int(m_gen.sum())
    c_bab = int(m_bab.sum())
    c_vx  = int(m_vx.sum())
    c_gk  = int(m_gk.sum())

    # Wallets that satisfy ALL thresholds simultaneously
    m_all = m_gen & m_bab & m_vx & m_gk
    c_all = int(m_all.sum())

    # KPIs + Highlight box
    st.markdown("**Results**")
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric(f"Wallets ∈ [{min_genesis}, {('∞' if max_genesis==0 else max_genesis)}] Genesis", fmt_int(c_gen))
    with k2: st.metric(f"Wallets ∈ [{min_babies}, {('∞' if max_babies==0 else max_babies)}] Babies", fmt_int(c_bab))
    with k3: st.metric(f"Wallets ∈ [{min_vx_any}, {('∞' if max_vx_any==0 else max_vx_any)}] VX (Any)", fmt_int(c_vx))
    with k4: st.metric(f"Wallets ∈ [{min_gk_any}, {('∞' if max_gk_any==0 else max_gk_any)}] Genkai (Any)", fmt_int(c_gk))

    st.markdown(
        f"""
        <div class="hi-box">
          <p class="hi-title">Meet <strong>all</strong> thresholds</p>
          <p class="hi-value">{c_all:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Matching wallets (all thresholds)**")
    cols_to_show = [GENESIS_COL, BABIES_COL, VX_ANY_COL, GK_ANY_COL]
    present_cols = ["wallet"] + [c for c in cols_to_show if c in holdings.columns]
    table = holdings.loc[m_all, present_cols].copy()

    # Sort by total across the four columns (desc)
    total_col = "_total_selected_"
    if len(present_cols) > 1:
        table[total_col] = table[[c for c in cols_to_show if c in table.columns]].sum(axis=1)
        table = table.sort_values(total_col, ascending=False).drop(columns=[total_col])

    # Optional filter
    q = st.text_input("Filter displayed wallets (substring)", value="")
    if q.strip():
        table = table[table["wallet"].str.lower().str.contains(q.strip().lower(), na=False)]

    st.dataframe(table.head(2000), width="stretch", height=460)

    # Download
    if not table.empty:
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        st.download_button(
            "Download matching wallets CSV",
            buf.getvalue().encode("utf-8"),
            file_name="threshold_query_wallets.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.button("Download matching wallets CSV", disabled=True, width="stretch")

# -------- Overview --------
with tab_overview:
    st.subheader("Wallet-holders per column")

    # Pick which column set to display
    all_cols = [c for c in holdings.columns if c != "wallet"]
    group_eth = [c for c in all_cols if "(Ronin)" not in c and "(Any)" not in c]
    group_ron = [c for c in all_cols if "(Ronin)" in c]
    group_any = [c for c in all_cols if "(Any)" in c]

    view = st.radio("Column set", ["All", "ETH only", "Ronin only", "Any-chain rollups"], horizontal=True)

    if view == "ETH only":
        cols_show = group_eth
    elif view == "Ronin only":
        cols_show = group_ron
    elif view == "Any-chain rollups":
        cols_show = group_any
    else:
        cols_show = all_cols

    totals = pd.DataFrame({
        "column": cols_show,
        "wallets_≥1": [int((holdings[c] > 0).sum()) for c in cols_show],
    }).sort_values("wallets_≥1", ascending=False).reset_index(drop=True)

    totals["share_of_wallets"] = (totals["wallets_≥1"] / max(1, distinct_wallets)).map(lambda x: f"{x:.1%}")

    fig = px.bar(totals, x="column", y="wallets_≥1", text="wallets_≥1")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=480, xaxis_tickangle=-25, yaxis_title="Wallets (≥1)")
    st.plotly_chart(fig, config={"responsive": True})

    st.dataframe(totals, width="stretch", height=320)

# -------- Downloads --------
with tab_downloads:
    st.subheader("Download CSVs")
    c1, c2 = st.columns(2)
    with c1:
        download_df_button(holdings, "per_wallet_holdings.csv", "per_wallet_holdings.csv")
    with c2:
        download_df_button(overlap, "overlap_summary.csv", "overlap_summary.csv")
        download_df_button(members, "combination_membership.csv", "combination_membership.csv")

st.caption("No API calls here — this app reads CSVs generated by your fetcher script.")
