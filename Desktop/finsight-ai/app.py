import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    .ai-card {
        background: linear-gradient(135deg, #1a1f35, #1e2540);
        border-left: 3px solid #4f8bf9;
        padding: 20px 24px;
        border-radius: 10px;
        margin-top: 8px;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c9d1d9;
        margin-bottom: 4px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .bullish  { background: #1a3a2a; color: #4caf82; }
    .bearish  { background: #3a1a1a; color: #ef5350; }
    .neutral  { background: #1a2a3a; color: #64b5f6; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # Bollinger Bands (20-day, 2σ)
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA20"] + 2 * std20
    df["BB_lower"] = df["SMA20"] - 2 * std20

    # RSI (14-period)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD
    df["MACD"]   = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal"]

    return df


# ─────────────────────────────────────────────
#  AI SUMMARY  ← swap body with LLM call later
# ─────────────────────────────────────────────

def generate_ai_summary(ticker: str, df: pd.DataFrame) -> str:
    """
    Rule-based market commentary.
    TO UPGRADE: replace this function body with an LLM API call.
    See README.md → "Upgrading to a real LLM" for the exact snippet.
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    close      = float(latest["Close"])
    rsi        = float(latest["RSI"])
    sma20      = float(latest["SMA20"])
    sma50      = float(latest["SMA50"])
    bb_upper   = float(latest["BB_upper"])
    bb_lower   = float(latest["BB_lower"])
    macd       = float(latest["MACD"])
    macd_hist  = float(latest["MACD_hist"])

    day_chg_pct = (close - float(prev["Close"])) / float(prev["Close"]) * 100
    ret_1m = ((close - float(df["Close"].iloc[-22])) / float(df["Close"].iloc[-22]) * 100
               if len(df) >= 23 else 0)
    ret_3m = ((close - float(df["Close"].iloc[-66])) / float(df["Close"].iloc[-66]) * 100
               if len(df) >= 67 else 0)

    # Signals
    trend     = "bullish" if close > sma50 else "bearish"
    rsi_label = ("overbought ⚠️" if rsi > 70
                 else "oversold 🔄" if rsi < 30
                 else "neutral ✅")
    bb_pos    = ("near the upper Bollinger Band, hinting at short-term resistance"
                 if close > bb_upper * 0.98
                 else "near the lower Bollinger Band, suggesting potential support"
                 if close < bb_lower * 1.02
                 else "within the Bollinger Bands, showing normal volatility")
    macd_bias = "positive MACD crossover — momentum building" if macd_hist > 0 else "negative MACD histogram — momentum fading"
    ma_cross  = ("golden cross zone (SMA20 > SMA50)" if sma20 > sma50
                 else "death cross zone (SMA20 < SMA50)")

    lines = [
        f"**{ticker} — Market Intelligence Report**\n",
        f"{ticker} is currently **{trend}**, trading at **${close:.2f}** "
        f"({'↑' if day_chg_pct >= 0 else '↓'}{abs(day_chg_pct):.2f}% today). "
        f"The stock sits {ma_cross}.\n",
        f"**Momentum:** RSI stands at **{rsi:.1f}** ({rsi_label}). "
        f"The MACD shows a {macd_bias}.\n",
        f"**Volatility:** Price is {bb_pos}.\n",
        f"**Performance:** 1-month return {ret_1m:+.1f}%"
        + (f" | 3-month return {ret_3m:+.1f}%" if ret_3m != 0 else "") + ".\n",
        "---",
        "_💡 Upgrade tip: drop your API key into `.streamlit/secrets.toml` and swap `generate_ai_summary()` "
        "with the LLM snippet in README.md for live GPT/Claude-powered summaries._",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ FinSight Controls")
    st.markdown("---")

    raw_input = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, JPM, GS, GOOGL")
    tickers   = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
    selected  = st.selectbox("Primary ticker", tickers)

    period_map = {
        "1 Month": 30, "3 Months": 90,
        "6 Months": 180, "1 Year": 365, "2 Years": 730,
    }
    period = st.selectbox("Period", list(period_map.keys()), index=3)
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=period_map[period])

    st.markdown("---")
    st.markdown("**Chart overlays**")
    show_sma    = st.checkbox("SMA 20 / 50",      value=True)
    show_bb     = st.checkbox("Bollinger Bands",  value=True)
    show_volume = st.checkbox("Volume",           value=True)
    show_rsi    = st.checkbox("RSI (14)",         value=True)
    show_macd   = st.checkbox("MACD",             value=False)

    st.markdown("---")
    st.caption("Data: Yahoo Finance · Refreshes every 5 min")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

st.title("📈 FinSight AI")
st.caption("Real-time financial analytics · Technical indicators · AI-powered market insights")
st.markdown("---")


# ─────────────────────────────────────────────
#  FETCH + PROCESS
# ─────────────────────────────────────────────

with st.spinner(f"Fetching {selected}…"):
    df = fetch_data(selected, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

if df.empty:
    st.error(f"No data found for **{selected}**. Check the ticker symbol.")
    st.stop()

df = compute_indicators(df)
latest = df.iloc[-1]
prev   = df.iloc[-2]


# ─────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────

price     = float(latest["Close"])
day_chg   = float(latest["Close"]) - float(prev["Close"])
day_pct   = day_chg / float(prev["Close"]) * 100
wk_ret    = (price - float(df["Close"].iloc[-6]))  / float(df["Close"].iloc[-6])  * 100 if len(df) > 5  else 0
mo_ret    = (price - float(df["Close"].iloc[-23])) / float(df["Close"].iloc[-23]) * 100 if len(df) > 22 else 0
rsi_val   = float(latest["RSI"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price",         f"${price:.2f}",      f"{day_pct:+.2f}%")
c2.metric("RSI (14)",      f"{rsi_val:.1f}",
          "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
c3.metric("1-Week Return", f"{wk_ret:+.2f}%")
c4.metric("1-Month Return",f"{mo_ret:+.2f}%")
c5.metric("Volume",        f"{int(latest['Volume']):,}")

st.markdown("---")


# ─────────────────────────────────────────────
#  PRICE CHART
# ─────────────────────────────────────────────

n_rows = 1 + int(show_volume) + int(show_rsi) + int(show_macd)
heights = [0.55]
if show_volume: heights.append(0.15)
if show_rsi:    heights.append(0.18)
if show_macd:   heights.append(0.12)
# normalise
total = sum(heights)
heights = [h / total for h in heights]

fig = make_subplots(
    rows=n_rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.025, row_heights=heights,
)

# ── Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Price",
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

# ── SMA
if show_sma:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20",
                             line=dict(color="#ffb74d", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50",
                             line=dict(color="#7986cb", width=1.5)), row=1, col=1)

# ── Bollinger Bands
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                             line=dict(color="rgba(100,200,255,0.45)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                             line=dict(color="rgba(100,200,255,0.45)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(100,200,255,0.05)"), row=1, col=1)

cur = 2

# ── Volume
if show_volume:
    vol_colors = [
        "#26a69a" if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i]) else "#ef5350"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.65), row=cur, col=1)
    cur += 1

# ── RSI
if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#ab47bc", width=1.5)), row=cur, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)",  row=cur, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)", row=cur, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)",  line_width=0, row=cur, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.05)", line_width=0, row=cur, col=1)
    cur += 1

# ── MACD
if show_macd:
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],   name="MACD",
                             line=dict(color="#42a5f5", width=1.5)), row=cur, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal",
                             line=dict(color="#ef5350", width=1.2, dash="dot")), row=cur, col=1)
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram",
                         marker_color=hist_colors, opacity=0.6), row=cur, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.8)",
    xaxis_rangeslider_visible=False,
    height=620,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
)
fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  AI SUMMARY
# ─────────────────────────────────────────────

st.markdown("### 🤖 AI Market Summary")
with st.spinner("Generating insights…"):
    summary = generate_ai_summary(selected, df)

st.markdown(f'<div class="ai-card">{summary}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MULTI-STOCK COMPARISON
# ─────────────────────────────────────────────

if len(tickers) > 1:
    st.markdown("---")
    st.markdown("### 📊 Normalised Return Comparison")
    st.caption("All tickers rebased to 0% at the start of the selected period.")

    cmp_fig = go.Figure()
    for t in tickers:
        try:
            cdf = fetch_data(t, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            if not cdf.empty and len(cdf) > 1:
                norm = (cdf["Close"] / float(cdf["Close"].iloc[0]) - 1) * 100
                cmp_fig.add_trace(go.Scatter(x=cdf.index, y=norm, name=t, mode="lines"))
        except Exception:
            pass

    cmp_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        yaxis_title="Return (%)",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    cmp_fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    cmp_fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(cmp_fig, use_container_width=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption("📌 Data sourced via Yahoo Finance · For educational and portfolio purposes only · Not financial advice.")
