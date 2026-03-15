# 📈 FinSight AI

> Real-time financial analytics dashboard with AI-powered market insights — built with Python, Streamlit, and Plotly.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Live market data** | Real-time OHLCV data for any ticker via `yfinance` |
| **Technical indicators** | SMA 20/50, Bollinger Bands, RSI (14), MACD — computed from scratch |
| **AI market summary** | Rule-based commentary engine (LLM-upgradeable — see below) |
| **Multi-stock comparison** | Normalised return chart across multiple tickers |
| **Interactive dashboard** | Candlestick + volume + indicator subplots via Plotly |
| **Deployed** | One-click cloud deployment on Streamlit Cloud |

---

## 🛠 Tech Stack

- **Python** · Pandas · NumPy
- **Streamlit** — dashboard framework
- **Plotly** — interactive financial charts
- **yfinance** — Yahoo Finance data ingestion
- **Streamlit Cloud** — deployment

---

## ⚡ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud (5 minutes)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo · branch: `main` · file: `app.py`
4. Click **Deploy** — you'll get a live public URL

---

## 🤖 Upgrading to a Real LLM Summary

The `generate_ai_summary()` function in `app.py` currently uses rule-based logic.
To swap in a live LLM, replace the function body with:

```python
import anthropic  # or: import openai

def generate_ai_summary(ticker: str, df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    prompt = f"""
    Analyze {ticker}. Current price: ${float(latest['Close']):.2f}.
    RSI: {float(latest['RSI']):.1f}. 
    SMA20: ${float(latest['SMA20']):.2f}, SMA50: ${float(latest['SMA50']):.2f}.
    Write a 3-sentence professional market commentary.
    """
    # Anthropic version:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

Then add your key to `.streamlit/secrets.toml` (never commit this file):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# or
OPENAI_API_KEY = "sk-..."
```

On Streamlit Cloud, add the key under **App settings → Secrets**.

---

## 📁 Project Structure

```
finsight-ai/
├── app.py              # Main dashboard application
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes secrets & cache
└── README.md           # This file
```

---

## 📌 Disclaimer

This project is built for educational and portfolio purposes only. Nothing here constitutes financial advice.
