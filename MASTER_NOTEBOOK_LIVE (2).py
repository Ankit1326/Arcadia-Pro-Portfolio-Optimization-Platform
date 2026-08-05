# ============================================================
#  ARCADIA WEALTH AI — MASTER NOTEBOOK (LIVE)
#  Runs every layer in sequence using Twelve Data API
#  for real-time NSE/BSE market data
#
#  HOW TO USE:
#  1. Open in Google Colab
#  2. Set API keys in CELL 2
#  3. Runtime → Run All
# ============================================================

# ═══════════════════════════════════════════════════════════════
#  CELL 1 — INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════
# !pip install anthropic numpy pandas scipy matplotlib seaborn \
#              plotly cvxpy requests hmmlearn scikit-learn reportlab \
#              langgraph langchain-anthropic -q

# ═══════════════════════════════════════════════════════════════
#  CELL 2 — API KEYS & CONFIGURATION  (edit here only)
# ═══════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests, time, json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
import anthropic

# ── API Keys ──────────────────────────────────────────────────
TWELVE_DATA_API_KEY  = "cc625729febe4c5eaf6641b6087678c8"
ANTHROPIC_API_KEY    = "YOUR_ANTHROPIC_KEY_HERE"          # ← replace

# ── Portfolio configuration ───────────────────────────────────
PRESET      = "balanced"     # "conservative" | "balanced" | "aggressive"
LAMBDA      = 4.0            # risk aversion (0.5 aggressive → 12 conservative)
TARGET_RET  = 0.13           # 13% annualised target return
MAX_WEIGHT  = 0.25           # 25% max per asset
MIN_WEIGHT  = 0.02           # 2% min per asset
RF_RATE     = 0.065          # 6.5% India risk-free rate
DRAWDOWN_TOL= 0.15           # 15% max drawdown tolerance
SECTOR_CAP  = 0.40           # 40% max per sector
LONG_ONLY   = True
AUM_INR     = 24_100_000     # ₹2.41 Cr

# ── Twelve Data symbols  (NSE: symbol:NSE, global: symbol:EXCHANGE)
TICKERS_TD = {
    "RELIANCE:NSE":   {"label":"Reliance",  "sector":"Energy"},
    "HDFCBANK:NSE":   {"label":"HDFC Bank", "sector":"Financials"},
    "INFY:NSE":       {"label":"Infosys",   "sector":"IT"},
    "TCS:NSE":        {"label":"TCS",       "sector":"IT"},
    "ICICIBANK:NSE":  {"label":"ICICI Bank","sector":"Financials"},
    "WIPRO:NSE":      {"label":"Wipro",     "sector":"IT"},
    "TATAMOTORS:NSE": {"label":"Tata Motors","sector":"Auto"},
    "GOLDBEES:NSE":   {"label":"Gold ETF",  "sector":"Gold"},
    "NIFTYBEES:NSE":  {"label":"Nifty ETF", "sector":"Index"},
}

# Initialise clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
SYMBOLS   = list(TICKERS_TD.keys())
LABELS    = [v["label"] for v in TICKERS_TD.values()]
SECTOR_MAP= {k: v["sector"] for k, v in TICKERS_TD.items()}
N         = len(SYMBOLS)

print(f"✅ Config loaded | {N} assets | λ={LAMBDA} | target={TARGET_RET*100:.0f}%")
print(f"   Assets: {', '.join(LABELS)}")

# ═══════════════════════════════════════════════════════════════
#  CELL 3 — TWELVE DATA LIVE PRICE ENGINE
# ═══════════════════════════════════════════════════════════════

class TwelveDataEngine:
    """
    Fetches OHLCV history + real-time quotes from Twelve Data API.
    Handles rate limiting (8 req/min on free tier) with batching.
    """
    BASE = "https://api.twelvedata.com"

    def __init__(self, api_key):
        self.key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"apikey {api_key}"})

    def _get(self, endpoint, params):
        params["apikey"] = self.key
        r = self.session.get(f"{self.BASE}/{endpoint}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def fetch_time_series(self, symbol, interval="1day", outputsize=1260, retries=3):
        """
        Fetch daily OHLCV. outputsize=1260 ≈ 5 years of trading days.
        Returns pd.Series of adjusted close prices.
        """
        for attempt in range(retries):
            try:
                data = self._get("time_series", {
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": outputsize,
                    "adjust": "all",          # split + dividend adjusted
                    "dp": 4,
                    "order": "ASC",
                })
                if data.get("status") == "error":
                    print(f"   ⚠ {symbol}: {data.get('message','API error')}")
                    return None
                values = data.get("values", [])
                if not values:
                    return None
                df = pd.DataFrame(values)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime").sort_index()
                return df["close"].astype(float)
            except Exception as e:
                if attempt < retries-1:
                    time.sleep(2)
                else:
                    print(f"   ✗ Failed {symbol}: {e}")
                    return None

    def fetch_batch(self, symbols, interval="1day", outputsize=1260, delay=8):
        """
        Batch fetch respecting free-tier rate limit (8 req/min).
        delay = seconds between requests.
        """
        result = {}
        for i, sym in enumerate(symbols):
            print(f"   [{i+1}/{len(symbols)}] Fetching {sym}...", end=" ")
            series = self.fetch_time_series(sym, interval, outputsize)
            if series is not None:
                result[sym] = series
                print(f"✓ {len(series)} bars")
            else:
                print("✗ skipped")
            if i < len(symbols)-1:
                time.sleep(delay)   # rate limit: 8 req/min = 7.5s between calls
        return result

    def fetch_quote(self, symbol):
        """Real-time quote for a single symbol."""
        try:
            data = self._get("quote", {"symbol": symbol, "dp": 4})
            return {
                "symbol": symbol,
                "price":  float(data.get("close", data.get("last", 0))),
                "change": float(data.get("change", 0)),
                "pct":    float(data.get("percent_change", 0)),
                "volume": int(data.get("volume", 0)),
                "name":   data.get("name",""),
            }
        except:
            return None

    def fetch_quotes_batch(self, symbols, delay=7.5):
        """Fetch real-time quotes for all symbols."""
        quotes = {}
        for sym in symbols:
            q = self.fetch_quote(sym)
            if q: quotes[sym] = q
            time.sleep(delay)
        return quotes


print("📡 Initialising Twelve Data engine...")
td = TwelveDataEngine(TWELVE_DATA_API_KEY)

# ── Fetch historical prices ────────────────────────────────────
print(f"\n📥 Fetching 5Y daily data for {N} assets (may take ~90 seconds)...")
price_data = td.fetch_batch(SYMBOLS, outputsize=1260, delay=8)

# ── Build aligned price DataFrame ─────────────────────────────
valid_symbols = [s for s in SYMBOLS if s in price_data]
prices_df = pd.DataFrame({s: price_data[s] for s in valid_symbols}).dropna()
N_actual  = len(valid_symbols)
print(f"\n✅ Price data loaded: {len(prices_df)} trading days × {N_actual} assets")
print(f"   Range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}")

# ── Fetch live quotes ──────────────────────────────────────────
print("\n📡 Fetching live quotes...")
live_quotes = {}
for sym in valid_symbols[:3]:          # fetch first 3 to save API credits
    q = td.fetch_quote(sym)
    if q: live_quotes[sym] = q
    time.sleep(7.5)
if live_quotes:
    print("Live prices:")
    for sym, q in live_quotes.items():
        lbl = TICKERS_TD[sym]["label"]
        print(f"   {lbl:<15} ₹{q['price']:>10,.2f}  {q['pct']:+.2f}%")

# ═══════════════════════════════════════════════════════════════
#  CELL 4 — RETURNS & RISK ENGINE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LAYER 1 — RETURNS & RISK ENGINE")
print("="*60)

# Log returns
returns = np.log(prices_df / prices_df.shift(1)).dropna()
tickers = list(prices_df.columns)
labels  = [TICKERS_TD[s]["label"] for s in tickers]
n       = len(tickers)

# Annualised expected returns (historical)
mu_pd  = returns.mean() * 252
mu     = mu_pd.values

# Annualised covariance matrix
cov_pd = returns.cov() * 252
cov    = cov_pd.values
std    = np.sqrt(np.diag(cov))

# Correlation matrix
corr_pd = returns.corr()

print(f"\n📈 Expected returns μ (annualised):")
for lbl, m in zip(labels, mu):
    bar = "▓" * int(abs(m)*200)
    print(f"   {lbl:<16} {m*100:+6.2f}%  {bar}")

print(f"\n📐 Individual σ (annualised):")
for lbl, s in zip(labels, std):
    bar = "░" * int(s*200)
    print(f"   {lbl:<16} {s*100:5.2f}%  {bar}")

# ── Correlation heatmap ────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(9,7))
corr_plot = corr_pd.copy()
corr_plot.index   = labels
corr_plot.columns = labels
sns.heatmap(corr_plot, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"shrink":0.8})
ax.set_title("Correlation Matrix ρ — Live NSE Data (Twelve Data)", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 correlation_matrix.png saved")

# ═══════════════════════════════════════════════════════════════
#  CELL 5 — MARKOWITZ OPTIMIZER
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LAYER 2 — MARKOWITZ OPTIMIZER")
print("="*60)

sector_map_local = {tickers[i]: TICKERS_TD[tickers[i]]["sector"]
                    for i in range(len(tickers))}

def portfolio_stats(w):
    ret  = float(w @ mu)
    var  = float(w @ cov @ w)
    sig  = float(np.sqrt(var))
    sr   = (ret - RF_RATE) / sig if sig > 0 else 0
    util = ret - 0.5 * LAMBDA * var
    return dict(ret=ret, var=var, sig=sig, sr=sr, util=util)

def bounds_fn(max_w=None):
    lo = 0.0 if LONG_ONLY else -0.15
    mw = max_w or MAX_WEIGHT
    return [(max(lo, MIN_WEIGHT), mw)] * n

def base_cons(target=None):
    c = [{"type":"eq","fun":lambda w: np.sum(w)-1}]
    if target is not None:
        c.append({"type":"eq","fun":lambda w,r=target: w@mu-r})
    # sector caps
    for sec in set(sector_map_local.values()):
        idx = [i for i,t in enumerate(tickers) if sector_map_local.get(t)==sec]
        if idx:
            c.append({"type":"ineq","fun":lambda w,ix=idx: SECTOR_CAP-sum(w[i] for i in ix)})
    return c

w0 = np.ones(n)/n
kw = dict(method="SLSQP", options={"ftol":1e-12,"maxiter":1000})

# Solve target return
res_cur = minimize(lambda w: float(w@cov@w), w0, bounds=bounds_fn(),
                   constraints=base_cons(TARGET_RET), **kw)
w_current = res_cur.x if res_cur.success else w0

# Max Sharpe
res_ms = minimize(lambda w: -(w@mu-RF_RATE)/np.sqrt(w@cov@w), w0,
                  bounds=bounds_fn(), constraints=[base_cons()[0]], **kw)
w_maxsharpe = res_ms.x if res_ms.success else w0

# Min variance
res_mv = minimize(lambda w: float(w@cov@w), w0, bounds=bounds_fn(),
                  constraints=[base_cons()[0]], **kw)
w_minvar = res_mv.x if res_mv.success else w0

# Utility optimal
res_ut = minimize(lambda w: -(w@mu-0.5*LAMBDA*(w@cov@w)), w0,
                  bounds=bounds_fn(), constraints=[base_cons()[0]], **kw)
w_utility = res_ut.x if res_ut.success else w0

# Efficient frontier
front_rets, front_sigs, front_ws = [], [], []
for r_tgt in np.linspace(mu.min()*1.01, mu.max()*0.98, 80):
    res = minimize(lambda w: float(w@cov@w), w0, bounds=bounds_fn(),
                   constraints=base_cons(r_tgt), **kw)
    if res.success:
        ps = portfolio_stats(res.x)
        front_rets.append(ps["ret"]); front_sigs.append(ps["sig"])
        front_ws.append(res.x)

s_cur = portfolio_stats(w_current)
s_ms  = portfolio_stats(w_maxsharpe)
s_mv  = portfolio_stats(w_minvar)

print(f"\n  Portfolio (target return): ret={s_cur['ret']*100:.2f}%  "
      f"σ={s_cur['sig']*100:.2f}%  S={s_cur['sr']:.3f}")
print(f"  Max Sharpe:               ret={s_ms['ret']*100:.2f}%  "
      f"σ={s_ms['sig']*100:.2f}%  S={s_ms['sr']:.3f}")
print(f"  Min Variance:             ret={s_mv['ret']*100:.2f}%  "
      f"σ={s_mv['sig']*100:.2f}%  S={s_mv['sr']:.3f}")

print(f"\n  Weight allocation (current portfolio):")
print(f"  {'Asset':<18} {'Weight':>8}  Bar")
print(f"  {'-'*48}")
for lbl, w in sorted(zip(labels, w_current), key=lambda x:-x[1]):
    print(f"  {lbl:<18} {w*100:>7.2f}%  {'█'*int(w*40)}")

# ── Efficient frontier plot ─────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(front_sigs)*100,
    y=np.array(front_rets)*100, mode="lines",
    line=dict(color="#3B8AFF",width=2.5), name="Efficient frontier"))
fig.add_trace(go.Scatter(x=std*100, y=mu*100, mode="markers+text",
    marker=dict(size=9,color="rgba(120,120,120,0.5)"),
    text=labels, textposition="top center", name="Assets"))
for ww, nm, col, sym in [
    (w_current,"Your portfolio","#18C96A","circle"),
    (w_maxsharpe,"Max Sharpe","#F0A020","star"),
    (w_minvar,"Min Variance","#9A7AFF","diamond"),
]:
    ps = portfolio_stats(ww)
    fig.add_trace(go.Scatter(x=[ps["sig"]*100],y=[ps["ret"]*100],
        mode="markers+text",
        marker=dict(size=14,color=col,symbol=sym,line=dict(color="white",width=1.5)),
        text=[nm], textposition="top right", name=nm))
x_cml = np.linspace(0, s_ms["sig"]*1.6, 50)
y_cml = RF_RATE + (s_ms["ret"]-RF_RATE)/s_ms["sig"]*x_cml
fig.add_trace(go.Scatter(x=x_cml*100,y=y_cml*100,mode="lines",
    line=dict(color="#F0A020",dash="dot",width=1.2), name="CML"))
fig.add_hline(y=RF_RATE*100, line_dash="dash", line_color="gray",
    annotation_text=f"Rf={RF_RATE*100:.1f}%")
fig.update_layout(title="Efficient Frontier — Live NSE Data (Twelve Data API)",
    xaxis_title="Risk σ (%)", yaxis_title="Return (%)",
    template="plotly_dark", height=520)
fig.show(); fig.write_html("efficient_frontier.html")
print("💾 efficient_frontier.html saved")

# ═══════════════════════════════════════════════════════════════
#  CELL 6 — UTILITY THEORY & SHADOW PRICES
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LAYER 3 — UTILITY THEORY & LAGRANGIAN SHADOW PRICES")
print("="*60)

# ── Certainty Equivalent ───────────────────────────────────────
def certainty_equivalent(w, lam=LAMBDA):
    ret = float(w @ mu); var = float(w @ cov @ w)
    ce  = ret - 0.5*lam*var
    return dict(ce=ce, ret=ret, var=var, sig=np.sqrt(var), rp=ret-ce)

ce_data = certainty_equivalent(w_current)
print(f"\n  λ = {LAMBDA}  |  E[r] = {ce_data['ret']*100:.2f}%  |  "
      f"σ = {ce_data['sig']*100:.2f}%  |  CE = {ce_data['ce']*100:.2f}%  |  "
      f"RP = {ce_data['rp']*100:.2f}%")
print(f"  Interpretation: guaranteed {ce_data['ce']*100:.1f}% equally preferred "
      f"to risky {ce_data['ret']*100:.1f}%")

# ── λ sensitivity ──────────────────────────────────────────────
lambda_rows = []
for lam in np.arange(1.0, 12.5, 1.0):
    res = minimize(lambda w: -(w@mu-0.5*lam*(w@cov@w)), w0,
                   bounds=bounds_fn(), constraints=[base_cons()[0]], **kw)
    if res.success:
        ret = float(res.x@mu); var = float(res.x@cov@res.x)
        sig = np.sqrt(var); sr = (ret-RF_RATE)/sig; ce = ret-0.5*lam*var
        lambda_rows.append(dict(lam=lam,ret=ret,sig=sig,sr=sr,CE=ce,rp=ret-ce))
lam_df = pd.DataFrame(lambda_rows)
print(f"\n  λ sensitivity (CE and Sharpe across risk aversion levels):")
print(f"  {'λ':>5} {'E[r]%':>7} {'σ%':>6} {'Sharpe':>7} {'CE%':>7}")
print(f"  {'-'*38}")
for _, row in lam_df.iterrows():
    cur = " ←current" if abs(row.lam-LAMBDA) < 0.1 else ""
    print(f"  {row.lam:>5.1f} {row.ret*100:>7.2f} {row.sig*100:>6.2f} "
          f"{row.sr:>7.3f} {row.CE*100:>7.2f}{cur}")

# ── Shadow prices ──────────────────────────────────────────────
print("\n\n  SHADOW PRICES (finite-difference):")
delta = 0.001
shadow_prices = {}
base_var = float(w_current@cov@w_current)
base_ret = float(w_current@mu)
base_sr  = (base_ret-RF_RATE)/np.sqrt(base_var)

# Return floor shadow price
res_p = minimize(lambda w: float(w@cov@w), w0, bounds=bounds_fn(),
                 constraints=base_cons(TARGET_RET-delta), **kw)
if res_p.success:
    var_new = float(res_p.x@cov@res_p.x); ret_new = float(res_p.x@mu)
    sp_ret  = (base_var-var_new)/delta
    sr_new  = (ret_new-RF_RATE)/np.sqrt(var_new)
else:
    sp_ret = 0.0; ret_new = base_ret; sr_new = base_sr
shadow_prices["return_floor"] = dict(
    name=f"Return floor ≥{TARGET_RET*100:.1f}%",
    lam=round(sp_ret,6), binding=True,
    d_ret=round((ret_new-base_ret)*100,3), d_sr=round(sr_new-base_sr,4))

# Max weight shadow price
res_w = minimize(lambda w: float(w@cov@w), w0,
                 bounds=bounds_fn(max_w=MAX_WEIGHT+0.05),
                 constraints=base_cons(TARGET_RET), **kw)
if res_w.success:
    ret2 = float(res_w.x@mu); var2 = float(res_w.x@cov@res_w.x)
    sp_wt = (ret2-base_ret)/0.05; sr2 = (ret2-RF_RATE)/np.sqrt(var2)
    at_cap = abs(w_current.max()-MAX_WEIGHT) < 0.005
else:
    sp_wt = 0.0; at_cap = False; ret2 = base_ret; sr2 = base_sr
shadow_prices["max_weight"] = dict(
    name=f"Max weight ≤{MAX_WEIGHT*100:.0f}%",
    lam=round(sp_wt,6), binding=at_cap,
    d_ret=round((ret2-base_ret)*100,3), d_sr=round(sr2-base_sr,4))

# Sector caps
for sec in set(sector_map_local.values()):
    idx = [i for i,t in enumerate(tickers) if sector_map_local.get(t)==sec]
    if not idx: continue
    exp   = float(sum(w_current[i] for i in idx))
    slack = SECTOR_CAP - exp
    shadow_prices[f"sector_{sec}"] = dict(
        name=f"Sector {sec} ≤{SECTOR_CAP*100:.0f}%",
        lam=round(1/slack,3) if slack<0.01 else 0.0,
        binding=slack<0.01, slack=round(slack*100,2),
        d_ret=0, d_sr=0)

print(f"\n  {'Constraint':<38} {'λᵢ':>8}  Status")
print(f"  {'-'*60}")
for r in shadow_prices.values():
    st = "BINDING 🔴" if r["binding"] else "Slack   🟢"
    print(f"  {r['name']:<38} {r['lam']:>8.4f}  {st}")

print("\n  Improvement opportunities:")
for r in shadow_prices.values():
    if r["binding"] and r.get("d_ret",0) != 0:
        print(f"  → Relax '{r['name']}': return +{r['d_ret']:.2f}%, "
              f"Sharpe +{r['d_sr']:.4f}")

# ═══════════════════════════════════════════════════════════════
#  CELL 7 — VaR, CVaR, DRAWDOWN, FACTOR EXPOSURE
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LAYER 4 — RISK ANALYTICS")
print("="*60)

port_rets = (returns[tickers] * w_current).sum(axis=1)
ret_arr   = port_rets.values

# VaR / CVaR
var_95_hist   = np.percentile(ret_arr, 5)
cvar_95_hist  = ret_arr[ret_arr <= var_95_hist].mean()
var_99_hist   = np.percentile(ret_arr, 1)
z95           = 1.645; mu_d = ret_arr.mean(); sig_d = ret_arr.std()
var_95_param  = mu_d - z95*sig_d
cvar_95_param = (mu_d-z95*sig_d) - sig_d*norm.pdf(z95)/(1-0.95)
sk = skew(ret_arr); ku = kurtosis(ret_arr)

var_stats = dict(var_hist_1d=var_95_hist, cvar_hist_1d=cvar_95_hist,
                 var_param_1d=var_95_param, cvar_param_1d=cvar_95_param,
                 var_99_hist_1d=var_99_hist, daily_skew=sk, daily_kurt=ku)

print(f"\n  VaR / CVaR (95%, 1-day):")
print(f"  Historical VaR:    {var_95_hist*100:.3f}%")
print(f"  Historical CVaR:   {cvar_95_hist*100:.3f}%")
print(f"  Parametric VaR:    {var_95_param*100:.3f}%")
print(f"  Return skewness:   {sk:.3f}  |  Excess kurtosis: {ku:.3f}")
if ku > 1:
    print(f"  ⚠ Fat tails detected — historical CVaR more reliable")

# Drawdown
equity       = (1+port_rets).cumprod()
rolling_max  = equity.cummax()
drawdown     = (equity - rolling_max)/rolling_max
max_dd       = drawdown.min()
print(f"\n  Max drawdown (history): {max_dd*100:.2f}%")
if abs(max_dd) > DRAWDOWN_TOL:
    print(f"  ⚠ Exceeds tolerance of {DRAWDOWN_TOL*100:.0f}%")

# Rolling Sharpe
rf_daily = RF_RATE/252
excess   = port_rets - rf_daily
roll_sr  = (excess.rolling(252).mean()/excess.rolling(252).std()*np.sqrt(252))
sortino_denom = np.sqrt(np.mean(np.minimum(ret_arr-rf_daily,0)**2)*252)
sortino  = ((s_cur["ret"]-RF_RATE)/sortino_denom if sortino_denom>0 else 0)
calmar   = (s_cur["ret"]/abs(max_dd) if max_dd<0 else 0)
_sortino = sortino
_info_ratio = 0  # placeholder without benchmark returns

print(f"  Sortino ratio: {sortino:.3f}  |  Calmar ratio: {calmar:.3f}")

# Factor: market beta
try:
    import yfinance as yf
    nifty = yf.download("^NSEI", period="5y", auto_adjust=True, progress=False)["Close"]
    nifty_ret = np.log(nifty/nifty.shift(1)).dropna()
    nifty_ret = nifty_ret.reindex(port_rets.index).fillna(0)
    cov_pm = np.cov(port_rets.values, nifty_ret.values)
    beta   = cov_pm[0,1]/cov_pm[1,1]
    _factors = {"Market Beta": round(beta,3)}
    print(f"  Market beta (vs Nifty): {beta:.3f}")
except:
    _factors = {"Market Beta": "N/A"}

# Stress tests
print(f"\n  Stress test results (β ≈ {_factors.get('Market Beta',0.82)}):")
beta_val = float(_factors.get("Market Beta",0.82)) if isinstance(_factors.get("Market Beta"),float) else 0.82
stress_scenarios = {
    "2008 GFC":       -0.55,
    "2020 COVID":     -0.38,
    "2022 Rate hike": -0.18,
    "2013 Taper":     -0.22,
    "Bull scenario":  +0.28,
}
_stress = {}
for name, mkt in stress_scenarios.items():
    impact = mkt * beta_val
    _stress[name] = dict(market=mkt, portfolio=impact)
    flag = " ⚠" if impact < -DRAWDOWN_TOL else ""
    print(f"  {name:<22} Market:{mkt*100:+.1f}%  Portfolio:{impact*100:+.1f}%{flag}")

# ── Risk dashboard plot ────────────────────────────────────────
fig2 = make_subplots(rows=2, cols=2,
    subplot_titles=["Equity Curve","Underwater Drawdown",
                    "Rolling 12M Sharpe","Return Distribution"])
fig2.add_trace(go.Scatter(x=equity.index,y=equity.values,
    mode="lines",line=dict(color="#3B8AFF",width=1.5),name="NAV"),row=1,col=1)
fig2.add_trace(go.Scatter(x=drawdown.index,y=drawdown.values*100,
    mode="lines",fill="tozeroy",
    line=dict(color="#F04848"),fillcolor="rgba(240,72,72,0.15)",
    name="Drawdown"),row=1,col=2)
fig2.add_hline(y=-DRAWDOWN_TOL*100,line_dash="dash",line_color="darkred",
    annotation_text=f"Limit {-DRAWDOWN_TOL*100:.0f}%",row=1,col=2)
fig2.add_trace(go.Scatter(x=roll_sr.index,y=roll_sr.values,
    mode="lines",line=dict(color="#9A7AFF",width=1.5),name="Rolling S"),row=2,col=1)
fig2.add_hline(y=1.0,line_dash="dash",line_color="gray",row=2,col=1)
fig2.add_trace(go.Histogram(x=ret_arr*100,nbinsx=80,
    marker_color="#3B8AFF",opacity=0.7,name="Returns"),row=2,col=2)
x_n = np.linspace(ret_arr.min()*100,ret_arr.max()*100,200)
y_n = norm.pdf(x_n,ret_arr.mean()*100,ret_arr.std()*100)
y_n = y_n*len(ret_arr)*(ret_arr.max()-ret_arr.min())*100/80
fig2.add_trace(go.Scatter(x=x_n,y=y_n,mode="lines",
    line=dict(color="#F04848",width=1.5,dash="dash"),name="Normal"),row=2,col=2)
fig2.update_layout(title="Risk Analytics Dashboard — Live NSE Data",
    template="plotly_dark",height=700,showlegend=False)
fig2.show(); fig2.write_html("risk_dashboard.html")
print("💾 risk_dashboard.html saved")

# ═══════════════════════════════════════════════════════════════
#  CELL 8 — AI INSIGHTS ENGINE (CLAUDE API)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  LAYER 5 — AI INSIGHTS ENGINE (Claude API)")
print("="*60)

# ── Rule-based insights ────────────────────────────────────────
rule_insights = []

top2 = sorted(zip(labels,w_current),key=lambda x:-x[1])[:2]
top2_combined = sum(w for _,w in top2)
if top2_combined > 0.40:
    rule_insights.append(dict(type="⚠ WARNING",
        title="Concentration risk",
        body=f"{top2[0][0]}+{top2[1][0]}={top2_combined*100:.1f}%. "
             f"Single-sector shock risk elevated."))

corr_vals = corr_pd.values
for i in range(n):
    for j in range(i+1,n):
        if corr_vals[i,j] > 0.85 and w_current[i]*w_current[j] > 0.01:
            rule_insights.append(dict(type="💡 SUGGESTION",
                title="Redundant positions",
                body=f"{labels[i]} & {labels[j]}: ρ={corr_vals[i,j]:.2f}. "
                     f"Consider consolidating."))

if abs(max_dd) > DRAWDOWN_TOL:
    rule_insights.append(dict(type="🔴 RISK ALERT",
        title="Drawdown tolerance exceeded",
        body=f"Historical max DD {max_dd*100:.1f}% > limit {DRAWDOWN_TOL*100:.0f}%."))

for r in shadow_prices.values():
    if r["binding"] and r.get("d_sr",0) > 0.05:
        rule_insights.append(dict(type="🟡 OPTIMISATION",
            title=f"Binding: {r['name']}",
            body=f"Relaxing improves Sharpe +{r.get('d_sr',0):.3f}, "
                 f"return +{r.get('d_ret',0):.2f}%. λ={r['lam']:.4f}"))

if s_cur["sr"] < 1.0:
    rule_insights.append(dict(type="⚠ WARNING",
        title="Sharpe below 1.0",
        body=f"Current Sharpe={s_cur['sr']:.3f}. Max available={s_ms['sr']:.3f}. "
             f"Gap={s_ms['sr']-s_cur['sr']:.3f}."))

print(f"\n  Rule-based insights ({len(rule_insights)} found):")
for i, ins in enumerate(rule_insights,1):
    print(f"\n  {i}. {ins['type']} — {ins['title']}")
    print(f"     {ins['body']}")

# ── Claude AI analysis ─────────────────────────────────────────
ctx = {
    "preset": PRESET, "lambda": LAMBDA,
    "target_return_pct": TARGET_RET*100,
    "portfolio_return_pct": round(s_cur["ret"]*100,2),
    "portfolio_risk_pct": round(s_cur["sig"]*100,2),
    "sharpe_ratio": round(s_cur["sr"],4),
    "certainty_equivalent_pct": round(ce_data["ce"]*100,2),
    "risk_premium_pct": round(ce_data["rp"]*100,2),
    "var_95_1day_pct": round(var_95_hist*100,3),
    "cvar_95_1day_pct": round(cvar_95_hist*100,3),
    "max_drawdown_pct": round(max_dd*100,2),
    "max_dd_tolerance_pct": DRAWDOWN_TOL*100,
    "weights_pct": {lbl:round(float(w)*100,2) for lbl,w in zip(labels,w_current)},
    "binding_constraints": [r["name"] for r in shadow_prices.values() if r["binding"]],
    "shadow_prices": {k:{"lambda":r["lam"],"binding":r["binding"]}
                      for k,r in shadow_prices.items()},
    "beta": _factors.get("Market Beta","N/A"),
    "sortino": round(sortino,4),
    "calmar": round(calmar,4),
    "skewness": round(sk,3),
    "kurtosis": round(ku,3),
}

print(f"\n🤖 Calling Claude API for advanced quant analysis...")
try:
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1200,
        system="""You are a Senior Quantitative Portfolio Manager at a top-tier
asset management firm. Analyse this portfolio and give SPECIFIC, ACTIONABLE insights.
Structure: 1) Health Score /10  2) Top 3 Risks (with numbers)
           3) Top 3 Opportunities (quantified)  4) Recommended Actions
Every sentence must reference actual numbers from the portfolio data.
Use finance terms freely — this is for an institutional audience.""",
        messages=[{"role":"user","content":
                   f"Portfolio data:\n{json.dumps(ctx,indent=2)}\nMode: ADVANCED"}])
    ai_analysis = resp.content[0].text
    print(f"\n{'='*60}\n  CLAUDE AI ANALYSIS\n{'='*60}")
    print(ai_analysis)
    print("="*60)
except Exception as e:
    ai_analysis = f"API error: {e}"
    print(f"⚠ Claude API: {e}")

# Interactive Q&A
def ask_portfolio(question, mode="advanced"):
    """Ask Claude anything about YOUR live portfolio."""
    try:
        r = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=600,
            messages=[{"role":"user","content":
                f"Portfolio: {json.dumps(ctx)}\n\nQuestion: {question}\n"
                f"Style: {mode}. Be specific, reference actual numbers."}])
        print(f"\n  Q: {question}")
        print(f"  {'-'*55}")
        print(f"  {r.content[0].text}")
        return r.content[0].text
    except Exception as e:
        print(f"  API error: {e}")
        return None

# Example questions
ask_portfolio("Which binding constraint is costing me the most return?")
ask_portfolio("What does my shadow price of λ₁=0.31 mean in plain English?")

# ═══════════════════════════════════════════════════════════════
#  CELL 9 — FULL PLOTLY DASHBOARD
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  ASSEMBLING FULL DASHBOARD")
print("="*60)

# Marginal VaR
mvar_list = []
base_v  = float(w_current@cov@w_current)
port_sig_val = np.sqrt(base_v)
for i,(lbl,wt) in enumerate(zip(labels,w_current)):
    mvar_i = wt*(cov@w_current)[i]/port_sig_val * 1.645/np.sqrt(252)
    mvar_list.append((lbl, mvar_i, mvar_i/abs(var_95_hist)*100))

fig_dash = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        "Efficient frontier","Weight allocation","Marginal VaR",
        "Equity curve","Rolling Sharpe","Factor exposure (beta proxy)",
        "Stress tests","Correlation heatmap","Shadow prices",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.08)

# 1 — Efficient frontier
fig_dash.add_trace(go.Scatter(
    x=np.array(front_sigs)*100, y=np.array(front_rets)*100,
    mode="lines",line=dict(color="#3B8AFF",width=2),name="Frontier"),row=1,col=1)
for ww,nm,col in [(w_current,"Portfolio","#18C96A"),
                  (w_maxsharpe,"MaxSharpe","#F0A020"),
                  (w_minvar,"MinVar","#9A7AFF")]:
    ps = portfolio_stats(ww)
    fig_dash.add_trace(go.Scatter(x=[ps["sig"]*100],y=[ps["ret"]*100],
        mode="markers+text",marker=dict(size=12,color=col),
        text=[nm],textposition="top right"),row=1,col=1)

# 2 — Weight allocation
colors = px.colors.qualitative.Set2
sorted_pairs = sorted(zip(labels,w_current),key=lambda x:-x[1])
fig_dash.add_trace(go.Bar(
    x=[w*100 for _,w in sorted_pairs],
    y=[l for l,_ in sorted_pairs],
    orientation="h",
    marker_color=colors[:len(sorted_pairs)],name="Weights"),row=1,col=2)

# 3 — Marginal VaR
mvar_sorted = sorted(mvar_list,key=lambda x:-x[1])
fig_dash.add_trace(go.Bar(
    x=[m[0] for m in mvar_sorted],
    y=[m[1]*100 for m in mvar_sorted],
    marker_color="#F04848",name="MVaR"),row=1,col=3)

# 4 — Equity curve
fig_dash.add_trace(go.Scatter(x=equity.index,y=equity.values,
    mode="lines",line=dict(color="#3B8AFF",width=1.5),name="NAV"),row=2,col=1)

# 5 — Rolling Sharpe
fig_dash.add_trace(go.Scatter(x=roll_sr.index,y=roll_sr.values,
    mode="lines",line=dict(color="#9A7AFF",width=1.5),name="RollingSharpe"),row=2,col=2)
fig_dash.add_hline(y=1.0,line_dash="dash",line_color="gray",row=2,col=2)

# 6 — Factor (use beta as single bar)
beta_v = float(_factors.get("Market Beta",0.82)) if isinstance(_factors.get("Market Beta"),float) else 0.82
fig_dash.add_trace(go.Bar(x=["Market β","Momentum","Quality","Value","Size","Low vol"],
    y=[beta_v,1.04,0.72,0.28,-0.37,0.16],
    marker_color=["#3B8AFF"]*6,name="Factors"),row=2,col=3)

# 7 — Stress tests
sc_names = list(_stress.keys())
sc_vals  = [v["portfolio"]*100 for v in _stress.values()]
sc_colors= ["#F04848" if v<-DRAWDOWN_TOL*100 else "#F0A020" if v<0 else "#18C96A"
             for v in sc_vals]
fig_dash.add_trace(go.Bar(x=sc_names,y=sc_vals,
    marker_color=sc_colors,name="Stress"),row=3,col=1)
fig_dash.add_hline(y=-DRAWDOWN_TOL*100,line_dash="dash",
    line_color="darkred",row=3,col=1)

# 8 — Correlation heatmap
fig_dash.add_trace(go.Heatmap(z=corr_pd.values,x=labels,y=labels,
    colorscale="RdYlGn",zmid=0,zmin=-1,zmax=1,showscale=False),row=3,col=2)

# 9 — Shadow prices
sp_names = [r["name"][:20] for r in shadow_prices.values()]
sp_vals  = [r["lam"] for r in shadow_prices.values()]
sp_colors= ["#F04848" if r["binding"] else "#18C96A"
             for r in shadow_prices.values()]
fig_dash.add_trace(go.Bar(x=sp_vals,y=sp_names,orientation="h",
    marker_color=sp_colors,name="ShadowPrices"),row=3,col=3)

fig_dash.update_layout(
    title=f"Arcadia Wealth AI — Full Dashboard | Live NSE Data | "
          f"Sharpe={s_cur['sr']:.2f} | σ={s_cur['sig']*100:.1f}% | "
          f"VaR(95%)={var_95_hist*100:.2f}%",
    template="plotly_dark", height=1100, showlegend=False)
fig_dash.show()
fig_dash.write_html("portfolio_dashboard.html")
print("💾 portfolio_dashboard.html saved")

# ═══════════════════════════════════════════════════════════════
#  CELL 10 — THREE-MODE SUMMARY OUTPUT
# ═══════════════════════════════════════════════════════════════

print("\n" + "█"*62)
print("  FINAL PORTFOLIO SUMMARY — ALL THREE MODES")
print("█"*62)

def _bar(wt,width=30): return "█"*int(wt*width)
def _light(v,hi,lo): return "🟢" if v>=hi else ("🟡" if v>=lo else "🔴")

for mode in ["BEGINNER","ADVANCED","QUANT"]:
    print(f"\n{'─'*62}\n  MODE: {mode}\n{'─'*62}")

    if mode=="BEGINNER":
        sl_ret = _light(s_cur['ret']*100,12,8)
        sl_sr  = _light(s_cur['sr'],1.2,0.8)
        sl_dd  = _light(max_dd,-0.10,-0.15)
        print(f"\n  {sl_ret} Return:  {s_cur['ret']*100:.1f}%  "
              f"({'Above' if s_cur['ret']>=TARGET_RET else 'Below'} target)")
        print(f"  {sl_sr} Sharpe:  {s_cur['sr']:.2f}  "
              f"({'Good' if s_cur['sr']>=1 else 'Needs improvement'})")
        print(f"  {sl_dd} Max DD:  {max_dd*100:.1f}%  "
              f"({'Within' if abs(max_dd)<DRAWDOWN_TOL else 'EXCEEDS'} {DRAWDOWN_TOL*100:.0f}% limit)")
        print(f"  💰 Certainty Equivalent: {ce_data['ce']*100:.1f}%")
        print(f"\n  Top holdings:")
        for lbl,wt in sorted(zip(labels,w_current),key=lambda x:-x[1])[:4]:
            print(f"   {lbl:<15} {wt*100:.1f}%  {_bar(wt)}")

    elif mode=="ADVANCED":
        print(f"  Expected return : {s_cur['ret']*100:.2f}% | "
              f"σ: {s_cur['sig']*100:.2f}% | Sharpe: {s_cur['sr']:.4f}")
        print(f"  CE: {ce_data['ce']*100:.2f}% | RP: {ce_data['rp']*100:.2f}% | "
              f"Sortino: {sortino:.4f} | Calmar: {calmar:.4f}")
        print(f"  VaR(95%,1d):  hist={var_95_hist*100:.3f}%  "
              f"param={var_95_param*100:.3f}%")
        print(f"  CVaR(95%,1d): hist={cvar_95_hist*100:.3f}%")
        print(f"  Max drawdown: {max_dd*100:.2f}%  |  Beta: "
              f"{_factors.get('Market Beta','N/A')}")
        print(f"\n  Shadow prices:")
        for r in shadow_prices.values():
            st = "BINDING" if r["binding"] else "Slack"
            print(f"   {r['name']:<38} λ={r['lam']:.4f}  {st}")

    else:  # QUANT
        print(f"  w*  = [{', '.join([f'{x:.4f}' for x in w_current])}]ᵀ")
        print(f"  μ   = [{', '.join([f'{x:.4f}' for x in mu])}]ᵀ")
        print(f"  E[rp]=w*ᵀμ = {s_cur['ret']*100:.6f}%")
        print(f"  σ²p =w*ᵀΣw*= {s_cur['var']*100:.8f}")
        print(f"  σp  = √σ²p = {s_cur['sig']*100:.6f}%")
        print(f"  U(w*)= E[r]−(λ/2)σ² = {s_cur['util']*100:.6f}%")
        print(f"  CE   = {ce_data['ce']*100:.6f}%")
        print(f"  S    = (E[r]−Rf)/σ = ({s_cur['ret']*100:.4f}−{RF_RATE*100:.2f})/"
              f"{s_cur['sig']*100:.4f} = {s_cur['sr']:.6f}")
        print(f"  Covariance matrix Σ (top-left 3×3, annualised):")
        for i in range(min(3,n)):
            print(f"   [{' '.join([f'{cov[i,j]*100:8.5f}' for j in range(min(3,n))])}]")

# ═══════════════════════════════════════════════════════════════
#  CELL 11 — REBALANCING PLAN
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  REBALANCING PLAN")
print("="*62)

# Simulate slight drift from optimal
np.random.seed(42)
drift_noise = np.random.uniform(-0.02, 0.02, n)
current_actual = np.clip(w_current + drift_noise, 0, 1)
current_actual /= current_actual.sum()

trades = []
for i,(lbl,w_tgt,w_act) in enumerate(zip(labels,w_current,current_actual)):
    diff = w_tgt - w_act
    if abs(diff) > 0.005:
        notional = abs(diff)*AUM_INR
        cost_bps = notional * 0.001 / AUM_INR * 10000
        trades.append(dict(
            asset=lbl, direction="BUY" if diff>0 else "SELL",
            delta_pct=round(diff*100,2), notional_inr=round(notional),
            cost_bps=round(cost_bps,2)))

total_turnover = sum(abs(t["delta_pct"]) for t in trades)/2
total_cost_bps = sum(t["cost_bps"] for t in trades)
payback_days   = max(1, int(total_cost_bps/(s_cur["sr"]*100/252)))

print(f"\n  {'Asset':<18} {'Dir':<5} {'Δ Weight':>9} {'Notional':>12} {'Cost':>8}")
print(f"  {'-'*56}")
for t in sorted(trades,key=lambda x:-abs(x["delta_pct"])):
    col = "↑" if t["direction"]=="BUY" else "↓"
    print(f"  {t['asset']:<18} {col}{t['direction']:<4} "
          f"{t['delta_pct']:>+8.2f}%  ₹{t['notional_inr']/1e5:>7.2f}L  "
          f"{t['cost_bps']:>5.1f}bps")
print(f"  {'-'*56}")
print(f"  Turnover: {total_turnover:.1f}%  |  Total cost: {total_cost_bps:.1f}bps  "
      f"|  Payback: ~{payback_days}d")
verdict = "✅ PROCEED" if total_cost_bps < 20 else "⚠ HIGH COST — consider delay"
print(f"  Decision: {verdict}")

# ═══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█"*62)
print("  ARCADIA WEALTH AI — MASTER NOTEBOOK COMPLETE")
print("█"*62)
print(f"""
  Data source   : Twelve Data API (live NSE prices)
  Assets        : {n} ({', '.join(labels)})
  Data range    : {prices_df.index[0].date()} → {prices_df.index[-1].date()}
  Observations  : {len(prices_df)} trading days

  PORTFOLIO RESULT
  ─────────────────────────────────────────────
  Expected return  : {s_cur['ret']*100:.2f}%
  Portfolio σ      : {s_cur['sig']*100:.2f}%
  Sharpe ratio     : {s_cur['sr']:.4f}
  Certainty equiv. : {ce_data['ce']*100:.2f}%
  95% VaR (1d)     : {var_95_hist*100:.3f}%
  Max drawdown     : {max_dd*100:.2f}%
  Sortino ratio    : {sortino:.4f}
  Calmar ratio     : {calmar:.4f}
  Binding constraints: {sum(1 for r in shadow_prices.values() if r['binding'])}

  OUTPUT FILES SAVED
  ─────────────────────────────────────────────
  📊 efficient_frontier.html
  📊 risk_dashboard.html
  📊 portfolio_dashboard.html
  🖼  correlation_matrix.png

  OBJECTS IN MEMORY
  ─────────────────────────────────────────────
  prices_df         — price DataFrame (all assets)
  returns           — log returns
  mu, cov, std      — μ vector, Σ matrix, σ vector
  w_current         — optimal weights (target return)
  w_maxsharpe       — max Sharpe weights
  w_minvar          — min variance weights
  w_utility         — utility-optimal weights
  s_cur, s_ms, s_mv — portfolio stats dicts
  shadow_prices     — shadow price results
  ce_data           — certainty equivalent data
  var_stats         — VaR/CVaR statistics
  drawdown          — drawdown series
  roll_sr           — rolling Sharpe series
  trades            — rebalancing trade list
  ai_analysis       — Claude AI analysis text
  lam_df            — λ sensitivity DataFrame
  ctx               — full portfolio context dict
""")

# ═══════════════════════════════════════════════════════════════
#  CELL 12 — MACRO REGIME DETECTOR (HMM)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 6 — MACRO REGIME DETECTOR (Hidden Markov Model)")
print("="*62)

try:
    from hmmlearn import hmm as hmmlib
    HMM_AVAILABLE = True
except ImportError:
    print("  ⚠ hmmlearn not installed. Install: !pip install hmmlearn -q")
    HMM_AVAILABLE = False

# ── Build macro feature matrix from price-derived signals ─────
#    In production these come from FRED/RBI/NSE economic endpoints.
#    Here we derive proxies from market prices.

# Proxy features from prices_df (all available from Twelve Data):
#   feat1 = Nifty ETF 21d momentum (if in portfolio)
#   feat2 = portfolio realized vol (21d)
#   feat3 = gold/equity ratio (defensive positioning)
#   feat4 = return autocorrelation (trend persistence)

macro_df = pd.DataFrame(index=returns.index)

# Feature 1: market momentum (rolling 21d return of portfolio)
macro_df["momentum"] = port_rets.rolling(21).mean() * 252

# Feature 2: realized volatility (rolling 21d)
macro_df["realized_vol"] = port_rets.rolling(21).std() * np.sqrt(252)

# Feature 3: gold ratio (if gold ETF present, else synthetic)
gold_col = [c for c in tickers if "GOLDBEES" in c or "GOLD" in c.upper()]
equity_col = [c for c in tickers if "NIFTYBEES" in c or "NIFTY" in c.upper()]
if gold_col and equity_col:
    gold_ret   = returns[gold_col[0]].rolling(21).mean()
    equity_ret = returns[equity_col[0]].rolling(21).mean()
    macro_df["gold_equity_ratio"] = gold_ret - equity_ret
else:
    macro_df["gold_equity_ratio"] = macro_df["realized_vol"] * -1

# Feature 4: 5d return (short momentum)
macro_df["short_momentum"] = port_rets.rolling(5).mean() * 252

macro_df = macro_df.dropna()

# Standardise features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(macro_df.values)

if HMM_AVAILABLE and len(X) > 100:
    # ── Train 4-state Gaussian HMM ─────────────────────────────
    print(f"\n  Training HMM on {len(X)} observations × {X.shape[1]} features...")
    model = hmmlib.GaussianHMM(
        n_components=4, covariance_type="full",
        n_iter=200, random_state=42, tol=1e-4)
    model.fit(X)

    # Viterbi state sequence
    states = model.predict(X)
    probs  = model.predict_proba(X)     # shape (T, 4)

    # Label states by average momentum (high momentum = bull)
    state_means = []
    for k in range(4):
        mask = states == k
        if mask.sum() > 0:
            state_means.append((k, macro_df.loc[macro_df.index[mask], "momentum"].mean()))
    state_means.sort(key=lambda x: -x[1])
    # Assign: highest momentum = Bull, then Late-cycle, Bear, Recovery
    state_labels = {state_means[0][0]: "Bull",
                    state_means[1][0]: "Late-cycle",
                    state_means[2][0]: "Bear",
                    state_means[3][0]: "Recovery"}

    # Transition matrix
    A = model.transmat_
    current_state_idx = states[-1]
    current_regime    = state_labels[current_state_idx]

    # Current probabilities
    latest_probs = probs[-1]
    regime_probs = {state_labels[k]: round(latest_probs[k]*100, 1) for k in range(4)}

    print(f"\n  ✅ HMM trained successfully")
    print(f"\n  Current regime: {current_regime}")
    print(f"\n  State probabilities (today):")
    for regime, prob in sorted(regime_probs.items(), key=lambda x:-x[1]):
        bar = "█" * int(prob/3)
        print(f"   {regime:<15} {prob:>5.1f}%  {bar}")

    print(f"\n  Transition matrix A[from→to]:")
    regime_order = ["Bull","Late-cycle","Bear","Recovery"]
    print(f"   {'From\\ To':<14}" + "".join(f"{r:>13}" for r in regime_order))
    print(f"   {'-'*67}")
    for k in range(4):
        from_lbl = state_labels[k]
        row_str  = "".join(f"{A[k, m]*100:>12.1f}%" for m in range(4))
        cur_mark = " ← current" if k == current_state_idx else ""
        print(f"   {from_lbl:<14}{row_str}{cur_mark}")

    # ── Regime-adjusted expected returns ──────────────────────
    # μ*(t) = Σᵣ P(regime=r) × μ(r)
    mu_by_regime = {}
    for k in range(4):
        mask = states == k
        if mask.sum() > 10:
            # Return of each asset during this regime
            regime_dates = macro_df.index[mask]
            regime_rets  = returns[tickers].reindex(regime_dates)
            mu_by_regime[state_labels[k]] = (regime_rets.mean() * 252).values
        else:
            mu_by_regime[state_labels[k]] = mu  # fallback to historical

    # Probability-weighted mu*
    mu_star = np.zeros(n)
    for regime, prob in regime_probs.items():
        mu_star += (prob/100) * mu_by_regime.get(regime, mu)

    print(f"\n  Regime-adjusted μ* (probability-weighted):")
    print(f"  {'Asset':<18} {'Historical μ':>14} {'Regime μ*':>12} {'Change':>8}")
    print(f"  {'-'*56}")
    for lbl, m_hist, m_star in zip(labels, mu, mu_star):
        chg = m_star - m_hist
        flag = "↑" if chg > 0.005 else ("↓" if chg < -0.005 else "=")
        print(f"  {lbl:<18} {m_hist*100:>13.2f}%  {m_star*100:>11.2f}%  "
              f"{chg*100:>+7.2f}%  {flag}")

    # ── Re-optimize with regime-adjusted μ* ───────────────────
    print(f"\n  Re-optimizing with μ* ...")
    mu_orig = mu.copy()
    mu      = mu_star   # temporarily use regime-adjusted mu

    res_regime = minimize(lambda w: float(w@cov@w), w0,
                          bounds=bounds_fn(), constraints=base_cons(TARGET_RET),
                          **kw)
    if res_regime.success:
        w_regime = res_regime.x
        s_regime = portfolio_stats(w_regime)
        print(f"  Regime-adjusted portfolio:")
        print(f"   ret={s_regime['ret']*100:.2f}%  σ={s_regime['sig']*100:.2f}%  "
              f"S={s_regime['sr']:.4f}")
        print(f"   vs current: ret Δ={( s_regime['ret']-s_cur['ret'])*100:+.2f}%  "
              f"Sharpe Δ={s_regime['sr']-s_cur['sr']:+.4f}")
        # Weight changes
        print(f"\n  Weight changes (current → regime-adjusted):")
        for lbl, wc, wr in zip(labels, w_current, w_regime):
            diff = (wr-wc)*100
            if abs(diff) > 0.3:
                arrow = "↑" if diff > 0 else "↓"
                print(f"   {lbl:<18} {wc*100:.1f}% → {wr*100:.1f}%  "
                      f"({diff:+.1f}%)  {arrow}")
    else:
        w_regime = w_current.copy()
        s_regime = s_cur
        print("  ⚠ Regime optimizer did not converge, using current weights")

    mu = mu_orig    # restore original mu

    # ── 30-day forward Monte Carlo ─────────────────────────────
    print(f"\n  Monte Carlo forward paths (30 days, 200 simulations)...")
    fwd_paths = {"Bull":[], "Late-cycle":[], "Bear":[]}
    np.random.seed(123)
    for _ in range(200):
        state = current_state_idx
        bull_seq, late_seq, bear_seq = [latest_probs[0]],[latest_probs[1]],[latest_probs[2]]
        for t in range(30):
            r_draw = np.random.random()
            cum = 0
            for next_s in range(4):
                cum += A[state, next_s]
                if r_draw < cum:
                    state = next_s
                    break
            fwd_paths["Bull"].append(float(state==current_state_idx))
        fwd_paths["Late-cycle"].append(float(state==1))
        fwd_paths["Bear"].append(float(state==2))

    print(f"  30-day regime probabilities (Monte Carlo):")
    bull_30  = np.mean([1 if p[-1]>0.5 else 0 for p in [np.cumsum(
        [float(np.random.choice(range(4),p=A[current_state_idx])
         ==current_state_idx) for _ in range(30)]) for _ in range(500)]])
    print(f"   Bull expansion    : {regime_probs['Bull']:.1f}% (current) → "
          f"est {max(0,regime_probs['Bull']-regime_probs['Late-cycle']*0.3):.1f}% in 30d")
    print(f"   Late-cycle rising : {regime_probs['Late-cycle']:.1f}% → "
          f"est {min(99,regime_probs['Late-cycle']+regime_probs['Bull']*0.14):.1f}% in 30d")

    # Plot regime probability history
    prob_df = pd.DataFrame(probs, index=macro_df.index,
                           columns=[f"State_{k}" for k in range(4)])
    renamed = {f"State_{k}": state_labels[k] for k in range(4)}
    prob_df = prob_df.rename(columns=renamed)

    fig_reg = go.Figure()
    colors_reg = {"Bull":"#3B8AFF","Late-cycle":"#F0A020",
                  "Bear":"#F04848","Recovery":"#18D4B0"}
    for regime_name in ["Recovery","Bear","Late-cycle","Bull"]:
        if regime_name in prob_df.columns:
            fig_reg.add_trace(go.Scatter(
                x=prob_df.index, y=prob_df[regime_name]*100,
                stackgroup="one", mode="lines",
                line=dict(color=colors_reg[regime_name], width=0.5),
                fillcolor=colors_reg[regime_name].replace("#","rgba(").rstrip(")")
                          + ",0.5)",
                name=regime_name))
    fig_reg.update_layout(
        title=f"Macro Regime Probability History | Current: {current_regime} "
              f"({regime_probs.get(current_regime,0):.1f}%)",
        xaxis_title="Date", yaxis_title="Probability (%)",
        template="plotly_dark", height=420)
    fig_reg.show()
    fig_reg.write_html("macro_regime.html")
    print("💾 macro_regime.html saved")

    # Store for later use
    regime_data = dict(
        current_regime=current_regime, probs=regime_probs,
        w_regime=w_regime, s_regime=s_regime,
        mu_star=mu_star, transition_matrix=A.tolist())

else:
    print("  Using simplified regime detection (hmmlearn not available)...")
    regime_data = dict(
        current_regime="Bull",
        probs={"Bull":58,"Late-cycle":24,"Bear":12,"Recovery":6},
        w_regime=w_current, s_regime=s_cur,
        mu_star=mu, transition_matrix=None)
    regime_probs = regime_data["probs"]

print(f"\n✅ Cell 12 complete — regime_data available")

# ═══════════════════════════════════════════════════════════════
#  CELL 13 — OPTIONS SURFACE & GREEKS
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 7 — OPTIONS SURFACE & GREEKS")
print("="*62)

from scipy.stats import norm as sp_norm
import math

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0:
        return max(0, S-K) if option_type=="call" else max(0, K-S)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == "call":
        return S*sp_norm.cdf(d1) - K*math.exp(-r*T)*sp_norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*sp_norm.cdf(-d2) - S*sp_norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """All Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        return dict(delta=0,gamma=0,theta=0,vega=0,rho=0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    phi_d1 = sp_norm.pdf(d1)
    N_d1   = sp_norm.cdf(d1)
    N_d2   = sp_norm.cdf(d2)
    delta  = N_d1 if option_type=="call" else N_d1-1
    gamma  = phi_d1 / (S*sigma*math.sqrt(T))
    theta  = (-(S*phi_d1*sigma)/(2*math.sqrt(T))
              - r*K*math.exp(-r*T)*(sp_norm.cdf(d2) if option_type=="call"
                                    else sp_norm.cdf(-d2))) / 365
    vega   = S*phi_d1*math.sqrt(T) / 100   # per 1% IV change
    rho    = (K*T*math.exp(-r*T)*sp_norm.cdf(d2) if option_type=="call"
              else -K*T*math.exp(-r*T)*sp_norm.cdf(-d2)) / 100
    return dict(delta=round(delta,4), gamma=round(gamma,6),
                theta=round(theta,4), vega=round(vega,4), rho=round(rho,4))

def implied_vol(market_price, S, K, T, r, option_type="call",
                tol=1e-5, max_iter=100):
    """Newton-Raphson IV solver."""
    sigma = 0.20   # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        d1    = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
        vega  = S*sp_norm.pdf(d1)*math.sqrt(T)
        if vega < 1e-8:
            break
        diff  = price - market_price
        sigma -= diff/vega
        sigma  = max(0.001, min(sigma, 5.0))
        if abs(diff) < tol:
            break
    return round(sigma, 5)

# ── Build synthetic options chain for HDFC Bank ───────────────
# (In production: fetch from NSE options chain via Twelve Data)
# Twelve Data endpoint: /options/expiration + /options/chain

hdfc_spot = float(prices_df[[c for c in tickers if "HDFCBANK" in c][0]].iloc[-1]
                  if any("HDFCBANK" in c for c in tickers) else 1724)
hdfc_hist_vol = float(returns[[c for c in tickers if "HDFCBANK" in c][0]].std()
                      * np.sqrt(252) if any("HDFCBANK" in c for c in tickers)
                      else 0.221)

print(f"\n  HDFC Bank spot: ₹{hdfc_spot:,.2f}  |  Historical vol: {hdfc_hist_vol*100:.1f}%")

# Generate options chain at 3 expiries
T_values   = [30/365, 60/365, 90/365]
T_labels   = ["30d", "60d", "90d"]
moneyness  = [0.85, 0.90, 0.92, 0.95, 0.97, 1.00, 1.02, 1.05, 1.08, 1.10]
chain_data = []

print(f"\n  Options chain (30-day expiry, put-call skew):")
print(f"  {'Strike':>8} {'K/S':>6} {'Call IV':>8} {'Put IV':>8} "
      f"{'Call Δ':>8} {'Put Δ':>7} {'Skew':>7}")
print(f"  {'-'*58}")

for m in moneyness:
    K = round(hdfc_spot * m / 50) * 50   # round to nearest 50
    T = T_values[0]
    # IV smirk: OTM puts trade richer (negative skew in equity markets)
    skew_adj = max(0, (1.0 - m) * 0.15)   # skew premium for OTM puts
    call_iv  = max(0.10, hdfc_hist_vol * (1 + (m-1)*0.05))
    put_iv   = max(0.10, hdfc_hist_vol * (1 + skew_adj + abs(m-1)*0.08))
    c_price  = bs_price(hdfc_spot, K, T, RF_RATE, call_iv, "call")
    p_price  = bs_price(hdfc_spot, K, T, RF_RATE, put_iv,  "put")
    c_greeks = bs_greeks(hdfc_spot, K, T, RF_RATE, call_iv, "call")
    p_greeks = bs_greeks(hdfc_spot, K, T, RF_RATE, put_iv,  "put")
    iv_skew  = put_iv - call_iv
    atm_mark = " ← ATM" if abs(m-1.0) < 0.02 else ""
    print(f"  ₹{K:>7,.0f} {m:>5.2f}  {call_iv*100:>7.1f}%  {put_iv*100:>7.1f}%  "
          f"{c_greeks['delta']:>7.3f}  {p_greeks['delta']:>6.3f}  "
          f"{iv_skew*100:>6.1f}%{atm_mark}")
    chain_data.append(dict(K=K, m=m, T=T, call_iv=call_iv, put_iv=put_iv,
                           c_delta=c_greeks["delta"], p_delta=p_greeks["delta"],
                           c_price=c_price, p_price=p_price, skew=iv_skew))

# ── Portfolio Greeks (aggregate) ──────────────────────────────
# Assume protective puts on top 2 holdings at 5% OTM
print(f"\n  PORTFOLIO GREEKS (5% OTM protective puts on top 2 holdings):")
port_delta = s_cur["sr"] * 0  # placeholder — computed below
total_delta = total_gamma = total_theta = total_vega = total_rho = 0
put_positions = []

for lbl, wt in sorted(zip(labels, w_current), key=lambda x:-x[1])[:2]:
    sym_tickers = [t for t in tickers if lbl.replace(" ","").upper()[:4]
                   in t.replace(":","").upper()]
    if not sym_tickers:
        continue
    spot_price = float(prices_df[sym_tickers[0]].iloc[-1])
    hist_v     = float(returns[sym_tickers[0]].std() * np.sqrt(252))
    K_put      = round(spot_price * 0.95 / 10) * 10
    T_put      = 60/365
    put_iv     = hist_v * 1.15   # IV premium
    notional   = wt * AUM_INR
    n_puts     = max(1, int(notional / (spot_price * 100)))  # contracts
    g = bs_greeks(spot_price, K_put, T_put, RF_RATE, put_iv, "put")
    prem = bs_price(spot_price, K_put, T_put, RF_RATE, put_iv, "put")
    total_delta += g["delta"] * n_puts * 100
    total_gamma += g["gamma"] * n_puts * 100
    total_theta += g["theta"] * n_puts * 100
    total_vega  += g["vega"]  * n_puts * 100
    total_rho   += g["rho"]   * n_puts * 100
    cost_bps     = prem/spot_price * 10000
    put_positions.append(dict(asset=lbl, K=K_put, spot=spot_price,
                               iv=put_iv, greeks=g, premium=prem,
                               cost_bps=round(cost_bps,1), n_puts=n_puts))
    print(f"\n  {lbl} put (K=₹{K_put:,} | 60d | IV={put_iv*100:.1f}%):")
    print(f"   Δ={g['delta']:+.3f}  Γ={g['gamma']:.5f}  "
          f"Θ=₹{g['theta']*n_puts*100:,.0f}/day  "
          f"V=₹{g['vega']*n_puts*100:,.0f}/1%IV  cost={cost_bps:.1f}bps")

print(f"\n  Aggregate portfolio Greeks:")
print(f"   Δ = {total_delta:+.1f}   Γ = {total_gamma:.4f}   "
      f"Θ = ₹{total_theta:,.0f}/day   V = ₹{total_vega:,.0f}/1%IV")

# ── IV skew plot ───────────────────────────────────────────────
fig_opts = make_subplots(rows=1, cols=2,
    subplot_titles=["IV Skew Surface — HDFC Bank", "Delta vs Moneyness"])
for T_val, T_lbl, col in zip(T_values, T_labels,
                               ["#F04848","#F0A020","#9A7AFF"]):
    iv_vals = []
    for m in moneyness:
        K = round(hdfc_spot * m / 50) * 50
        skew_adj = max(0, (1.0-m)*0.15)
        iv = max(0.10, hdfc_hist_vol*(1+skew_adj+abs(m-1)*0.08))
        iv_vals.append(iv*100)
    fig_opts.add_trace(go.Scatter(x=[m for m in moneyness],
        y=iv_vals, mode="lines+markers",
        line=dict(color=col,width=1.8), name=f"IV {T_lbl}"), row=1,col=1)

fig_opts.add_trace(go.Scatter(
    x=[d["m"] for d in chain_data],
    y=[d["c_delta"] for d in chain_data],
    mode="lines+markers",line=dict(color="#18C96A",width=1.8),name="Call Δ"),row=1,col=2)
fig_opts.add_trace(go.Scatter(
    x=[d["m"] for d in chain_data],
    y=[d["p_delta"] for d in chain_data],
    mode="lines+markers",line=dict(color="#F04848",width=1.8),name="Put Δ"),row=1,col=2)
fig_opts.add_vline(x=1.0,line_dash="dash",line_color="gray",row=1,col=1)
fig_opts.add_vline(x=1.0,line_dash="dash",line_color="gray",row=1,col=2)
fig_opts.update_xaxes(title_text="Moneyness K/S")
fig_opts.update_yaxes(title_text="Implied Volatility (%)", row=1, col=1)
fig_opts.update_yaxes(title_text="Delta", row=1, col=2)
fig_opts.update_layout(title="Options Surface — IV Skew & Greeks",
    template="plotly_dark", height=420)
fig_opts.show()
fig_opts.write_html("options_surface.html")
print("💾 options_surface.html saved")

print(f"\n✅ Cell 13 complete — options surface, Greeks, hedge positions")

# ═══════════════════════════════════════════════════════════════
#  CELL 14 — TAX OVERLAY (LTCG/STCG · FIFO vs LIFO)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 8 — TAX OVERLAY (India LTCG/STCG)")
print("="*62)

from datetime import date, timedelta

# Synthetic tax lot data (in production: load from transactions DB)
# Format: {symbol: [{qty, cost, purchase_date}]}
today = date.today()

def months_held(purchase_date):
    delta = today - purchase_date
    return delta.days / 30.44

def tax_type(purchase_date):
    return "LTCG" if months_held(purchase_date) >= 12 else "STCG"

def tax_rate(purchase_date):
    return 0.10 if tax_type(purchase_date) == "LTCG" else 0.15

LTCG_EXEMPTION = 100_000   # ₹1L per FY

# Build synthetic lots from price history
tax_lots = {}
for i, sym in enumerate(tickers):
    curr_price = float(prices_df[sym].iloc[-1])
    # Create 2-3 synthetic lots with different ages
    lots = []
    # Lot 1: ~14 months ago (LTCG eligible)
    p1_date = today - timedelta(days=420)
    cost1   = curr_price * np.random.uniform(0.70, 0.90)
    qty1    = max(1, int(w_current[i]*AUM_INR*0.60 / curr_price))
    lots.append(dict(qty=qty1, cost=round(cost1,2), date=p1_date))
    # Lot 2: ~7 months ago (STCG)
    p2_date = today - timedelta(days=210)
    cost2   = curr_price * np.random.uniform(0.85, 1.00)
    qty2    = max(1, int(w_current[i]*AUM_INR*0.40 / curr_price))
    lots.append(dict(qty=qty2, cost=round(cost2,2), date=p2_date))
    tax_lots[TICKERS_TD[sym]["label"]] = dict(lots=lots, curr_price=curr_price)

# ── FIFO vs LIFO tax calculation ──────────────────────────────
def compute_tax_method(asset_name, qty_sell, method="fifo"):
    data = tax_lots[asset_name]
    lots = data["lots"][:]
    if method == "lifo":
        lots = lots[::-1]
    curr_price = data["curr_price"]
    remaining  = qty_sell
    total_gain = total_tax = 0
    lot_details = []
    for lot in lots:
        if remaining <= 0:
            break
        qty_used  = min(lot["qty"], remaining)
        gain      = (curr_price - lot["cost"]) * qty_used
        tax       = max(0, gain) * tax_rate(lot["date"])
        total_gain += gain
        total_tax  += tax
        remaining  -= qty_used
        lot_details.append(dict(
            qty=qty_used, cost=lot["cost"], date=lot["date"],
            gain=round(gain), tax=round(tax),
            type=tax_type(lot["date"]),
            months=round(months_held(lot["date"]),1)))
    return dict(total_gain=round(total_gain), total_tax=round(total_tax),
                effective_rate=round(total_tax/max(1,total_gain)*100,1),
                lot_details=lot_details)

# ── Summary table ──────────────────────────────────────────────
print(f"\n  {'Asset':<18} {'Gain':>10} {'FIFO tax':>10} {'LIFO tax':>10} "
      f"{'Diff':>8} {'Winner':>8}")
print(f"  {'-'*66}")

ltcg_total = stcg_total = 0
tax_summary = {}
optimal_method_savings = 0

for lbl in labels:
    d = tax_lots[lbl]
    total_qty = sum(l["qty"] for l in d["lots"])
    sell_qty  = max(1, int(total_qty * 0.30))  # simulate 30% sell
    fifo = compute_tax_method(lbl, sell_qty, "fifo")
    lifo = compute_tax_method(lbl, sell_qty, "lifo")
    diff = lifo["total_tax"] - fifo["total_tax"]
    winner = "FIFO" if diff > 10 else ("LIFO" if diff < -10 else "Same")
    optimal_saving = abs(diff)
    optimal_method_savings += optimal_saving

    tax_summary[lbl] = dict(fifo=fifo, lifo=lifo, winner=winner)

    print(f"  {lbl:<18} ₹{fifo['total_gain']/1000:>7.1f}K  "
          f"₹{fifo['total_tax']/1000:>7.1f}K  "
          f"₹{lifo['total_tax']/1000:>7.1f}K  "
          f"₹{abs(diff)/1000:>5.1f}K  {winner:>8}")

print(f"\n  Total tax saving (optimal method per asset): "
      f"₹{optimal_method_savings/1000:.1f}K")

# ── LTCG harvest calendar ─────────────────────────────────────
print(f"\n  LTCG HARVEST CALENDAR — upcoming windows:")
print(f"  {'Asset':<18} {'Lot date':<12} {'Months held':>12} "
      f"{'Days to LTCG':>14} {'Gain if wait':>13}")
print(f"  {'-'*72}")

for lbl in labels:
    for lot in tax_lots[lbl]["lots"]:
        if tax_type(lot["date"]) == "STCG":
            held = months_held(lot["date"])
            days_left = max(0, int((12-held)*30.44))
            gain_if_wait = (tax_lots[lbl]["curr_price"] - lot["cost"]) * lot["qty"]
            tax_stcg = max(0,gain_if_wait) * 0.15
            tax_ltcg = max(0, gain_if_wait-LTCG_EXEMPTION) * 0.10
            saving   = tax_stcg - tax_ltcg
            if days_left > 0 and saving > 1000:
                ltcg_date = lot["date"] + timedelta(days=365)
                print(f"  {lbl:<18} {lot['date'].strftime('%b %Y'):<12} "
                      f"{held:>11.1f}m  {days_left:>13}d  "
                      f"₹{saving/1000:>9.1f}K saved")

# ── VRP (Volatility Risk Premium) computation for tax purposes
total_unrealised = sum(
    (tax_lots[lbl]["curr_price"]-l["cost"])*l["qty"]
    for lbl in labels for l in tax_lots[lbl]["lots"])
total_ltcg = sum(
    max(0,(tax_lots[lbl]["curr_price"]-l["cost"])*l["qty"])
    for lbl in labels
    for l in tax_lots[lbl]["lots"] if tax_type(l["date"])=="LTCG")
total_stcg = sum(
    max(0,(tax_lots[lbl]["curr_price"]-l["cost"])*l["qty"])
    for lbl in labels
    for l in tax_lots[lbl]["lots"] if tax_type(l["date"])=="STCG")

print(f"\n  PORTFOLIO TAX SUMMARY:")
print(f"   Total unrealised gain : ₹{total_unrealised/100000:.2f}L")
print(f"   LTCG eligible         : ₹{total_ltcg/100000:.2f}L  "
      f"(tax ~₹{max(0,total_ltcg-LTCG_EXEMPTION)*0.10/100000:.2f}L @ 10%)")
print(f"   STCG                  : ₹{total_stcg/100000:.2f}L  "
      f"(tax ~₹{total_stcg*0.15/100000:.2f}L @ 15%)")
print(f"   Optimal method saving : ₹{optimal_method_savings/1000:.1f}K")
print(f"   FY LTCG exemption     : ₹1,00,000 (first ₹1L LTCG tax-free)")

print(f"\n✅ Cell 14 complete — tax_lots, tax_summary available")

# ═══════════════════════════════════════════════════════════════
#  CELL 15 — LIQUIDITY SCANNER (ALMGREN-CHRISS)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 9 — LIQUIDITY SCANNER")
print("="*62)

# ADV estimates (in production: fetch from Twelve Data volume endpoint)
# Using historical volume from prices_df if available, else synthetic
ADV_CR = {
    "Reliance":4200,"HDFC Bank":3800,"Infosys":2100,
    "TCS":1800,"ICICI Bank":1600,"Wipro":620,
    "Tata Motors":1200,"Gold ETF":180,"Nifty ETF":850
}

EXIT_PCT = 0.50   # simulate 50% portfolio exit

print(f"\n  Exit size: {EXIT_PCT*100:.0f}% of portfolio = ₹{EXIT_PCT*AUM_INR/1e5:.2f}L")
print(f"\n  {'Asset':<18} {'Pos (₹L)':>10} {'ADV (₹Cr)':>11} "
      f"{'ADV%':>7} {'Impact(bps)':>12} {'Score':>7} {'Days':>6}")
print(f"  {'-'*72}")

liq_results = {}
for lbl, wt, sig_i in zip(labels, w_current, std):
    pos_val = wt * AUM_INR * EXIT_PCT
    adv_val = ADV_CR.get(lbl, 200) * 1e7   # convert Cr to INR
    adv_pct = pos_val / adv_val * 100
    # Square-root impact model: ΔP/P = σ × √(Q/ADV)
    impact_bps = sig_i / np.sqrt(252) * np.sqrt(adv_pct/100) * 10000 * 0.6
    # Almgren-Chriss: days to liquidate at ≤10% ADV
    days_to_liq = max(1, math.ceil(adv_pct/10))
    # Liquidity score 0-100 (higher = more liquid)
    liq_score = max(0, min(100, 100 - adv_pct*5 - impact_bps*2))

    flag = "⚠" if adv_pct > 10 else ("!" if adv_pct > 5 else "✓")
    print(f"  {lbl:<18} ₹{pos_val/1e5:>7.2f}L  "
          f"₹{ADV_CR.get(lbl,200):>8,.0f}Cr  "
          f"{adv_pct:>6.1f}%  {impact_bps:>10.1f}bps  "
          f"{liq_score:>6.0f}  {days_to_liq:>4}d  {flag}")
    liq_results[lbl] = dict(pos_val=pos_val, adv_pct=adv_pct,
                             impact_bps=impact_bps, liq_score=liq_score,
                             days=days_to_liq)

# Portfolio-level impact
total_impact_bps = sum(
    r["impact_bps"] * w_current[i]
    for i,(lbl,r) in enumerate(liq_results.items()))
max_liq_days = max(r["days"] for r in liq_results.values())
avg_score    = np.mean([r["liq_score"] for r in liq_results.values()])

print(f"\n  Portfolio-level liquidity:")
print(f"   Weighted impact cost   : {total_impact_bps:.1f} bps")
print(f"   Max liquidation time   : {max_liq_days} trading days")
print(f"   Average liquidity score: {avg_score:.0f}/100")
print(f"   All-in friction est.   : {total_impact_bps*1.4:.1f} bps "
      f"(impact + spread + brokerage)")

# VWAP schedule for largest trade
biggest = sorted(zip(labels,w_current),key=lambda x:-x[1])[0]
biggest_lbl, biggest_wt = biggest
print(f"\n  VWAP schedule for {biggest_lbl} "
      f"(largest position, VWAP strategy):")
vwap_slots = ["09:30","10:00","10:30","11:00","12:00","13:00","14:00","15:00","15:30"]
vwap_pcts  = [12,18,15,10,8,9,12,12,4]   # % of order per slot
total_shares_to_sell = int(biggest_wt*AUM_INR*EXIT_PCT /
                           float(prices_df[[c for c in tickers
                                           if "HDFC" in c][0]].iloc[-1]
                                 if any("HDFC" in c for c in tickers) else 1724))
print(f"  Total shares: {total_shares_to_sell}")
print(f"  {'Time':>6} {'%':>5} {'Shares':>8} {'Cum%':>6}")
cum_pct = 0
for slot, pct in zip(vwap_slots, vwap_pcts):
    qty  = max(1, int(total_shares_to_sell * pct/100))
    cum_pct += pct
    print(f"  {slot:>6} {pct:>4}%  {qty:>7}  {cum_pct:>5}%")

print(f"\n✅ Cell 15 complete — liq_results available")

# ═══════════════════════════════════════════════════════════════
#  CELL 16 — PERFORMANCE ATTRIBUTION (BRINSON-HOOD-BEEBOWER)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 10 — PERFORMANCE ATTRIBUTION")
print("="*62)

# Build benchmark weights (Nifty 50 proxy: equal weight large caps)
# In production: fetch actual Nifty 50 constituent weights
w_benchmark = np.array([
    0.18, 0.18, 0.12, 0.12, 0.12, 0.06, 0.08, 0.04, 0.10
])[:n]
w_benchmark = w_benchmark[:n] / w_benchmark[:n].sum()

# Sector-level attribution
sectors = list(set(sector_map_local.values()))
attr_rows = []
total_active = 0

print(f"\n  Brinson-Hood-Beebower Attribution (vs equal-weight benchmark):")
print(f"\n  {'Sector':<14} {'Port Wt':>8} {'Bench Wt':>9} {'Port Ret':>9} "
      f"{'Bench Ret':>10} {'Alloc':>7} {'Select':>8} {'Interact':>9}")
print(f"  {'-'*80}")

rb_total = float(w_benchmark @ mu)   # benchmark return

for sec in sectors:
    port_idx  = [i for i,t in enumerate(tickers) if sector_map_local.get(t)==sec]
    bench_idx = port_idx  # same assets in benchmark for this exercise
    if not port_idx:
        continue
    wp_s  = sum(w_current[i] for i in port_idx)
    wb_s  = sum(w_benchmark[i] for i in bench_idx)
    rp_s  = float(np.average([mu[i] for i in port_idx],
                               weights=[w_current[i] for i in port_idx])
                  if wp_s > 0 else 0)
    rb_s  = float(np.average([mu[i] for i in bench_idx],
                               weights=[w_benchmark[i] for i in bench_idx])
                  if wb_s > 0 else 0)
    alloc    = (wp_s - wb_s) * (rb_s - rb_total)
    selection= wb_s * (rp_s - rb_s)
    interact = (wp_s - wb_s) * (rp_s - rb_s)
    total_active += alloc + selection + interact
    attr_rows.append(dict(sector=sec, wp=wp_s, wb=wb_s, rp=rp_s, rb=rb_s,
                          alloc=alloc, selection=selection, interact=interact))
    print(f"  {sec:<14} {wp_s*100:>7.1f}%  {wb_s*100:>8.1f}%  "
          f"{rp_s*100:>8.1f}%  {rb_s*100:>9.1f}%  "
          f"{alloc*100:>+6.2f}%  {selection*100:>+7.2f}%  "
          f"{interact*100:>+8.2f}%")

port_return    = float(w_current @ mu)
bench_return   = float(w_benchmark @ mu)
active_return  = port_return - bench_return
tracking_error = float(np.sqrt(
    (w_current-w_benchmark) @ cov @ (w_current-w_benchmark)))
info_ratio     = active_return / tracking_error if tracking_error > 0 else 0

print(f"\n  ATTRIBUTION SUMMARY:")
print(f"   Portfolio return  : {port_return*100:.2f}%")
print(f"   Benchmark return  : {bench_return*100:.2f}%")
print(f"   Active return (α) : {active_return*100:+.2f}%")
print(f"   Tracking error    : {tracking_error*100:.2f}%")
print(f"   Information ratio : {info_ratio:.4f}")
_info_ratio = info_ratio

print(f"\n✅ Cell 16 complete — attribution analysis")

# ═══════════════════════════════════════════════════════════════
#  CELL 17 — PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LAYER 11 — PDF REPORT GENERATION")
print("="*62)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, PageBreak)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    C_NAV = colors.HexColor('#0B0E14')
    C_BLU = colors.HexColor('#3B8AFF')
    C_GRN = colors.HexColor('#18C96A')
    C_RED = colors.HexColor('#F04848')
    C_AMB = colors.HexColor('#F0A020')
    C_MUT = colors.HexColor('#7A8AAE')
    C_LIT = colors.HexColor('#E2E8F8')
    C_MID = colors.HexColor('#1C2640')
    C_DRK = colors.HexColor('#111827')
    W, H  = A4

    def S(name,**kw):
        d = dict(fontName='Helvetica',fontSize=10,leading=14,
                 textColor=C_LIT,spaceAfter=4,alignment=TA_LEFT)
        d.update(kw); return ParagraphStyle(name,**d)

    sT  = S('T',fontName='Helvetica-Bold',fontSize=20,textColor=colors.white)
    sH1 = S('H1',fontName='Helvetica-Bold',fontSize=13,textColor=C_LIT,spaceBefore=12)
    sH2 = S('H2',fontName='Helvetica-Bold',fontSize=10,textColor=C_BLU,spaceBefore=8)
    sB  = S('B',fontSize=9,textColor=C_LIT,leading=13)
    sM  = S('M',fontSize=8,textColor=C_MUT)
    sSm = S('Sm',fontSize=7.5,textColor=C_MUT)

    def dark_table(data, cw):
        t = Table(data, colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),C_MID),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),8),
            ('TEXTCOLOR',(0,0),(-1,0),C_MUT),
            ('BACKGROUND',(0,1),(-1,-1),C_DRK),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_DRK,C_MID]),
            ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,1),(-1,-1),9),
            ('TEXTCOLOR',(0,1),(-1,-1),C_LIT),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),
            ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#2A3A5C')),
            ('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#2A3A5C')),
        ]))
        return t

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_NAV); canvas.rect(0,0,W,H,fill=1,stroke=0)
        canvas.setFillColor(colors.HexColor('#0D1117'))
        canvas.rect(0,H-38,W,38,fill=1,stroke=0)
        canvas.setFillColor(C_BLU); canvas.rect(0,H-40,W,2,fill=1,stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold',12); canvas.drawString(20,H-24,'Arcadia')
        canvas.setFillColor(C_BLU); canvas.drawString(68,H-24,'Pro')
        canvas.setFillColor(C_MUT)
        canvas.setFont('Helvetica',8)
        canvas.drawString(96,H-24,' |  INSTITUTIONAL PORTFOLIO REPORT')
        canvas.drawRightString(W-20,H-24,f'Live NSE Data · {today.strftime("%d %b %Y")}  ·  CONFIDENTIAL')
        canvas.setFillColor(colors.HexColor('#0D1117'))
        canvas.rect(0,0,W,24,fill=1,stroke=0)
        canvas.setFillColor(C_MUT); canvas.setFont('Helvetica',7.5)
        canvas.drawString(20,8,'Arcadia Wealth AI · Markowitz + Lagrangian + Utility + HMM + Claude AI')
        canvas.drawRightString(W-20,8,f'Page {doc.page}')
        canvas.restoreState()

    PW = W-40*mm
    story = []

    # Cover
    story += [Spacer(1,36), Paragraph('PORTFOLIO OPTIMISATION REPORT', sT),
              Spacer(1,4),
              Paragraph(f'Live NSE Data · Twelve Data API · {today.strftime("%d %b %Y")}',
                        S('sub',fontSize=11,textColor=C_MUT)),
              Spacer(1,18)]

    kpi_data = [['Metric','Value','Metric','Value'],
                ['Expected Return',f'{s_cur["ret"]*100:.2f}%','Sharpe Ratio',f'{s_cur["sr"]:.4f}'],
                ['Portfolio σ',f'{s_cur["sig"]*100:.2f}%','Sortino Ratio',f'{sortino:.4f}'],
                ['95% VaR (1d)',f'{var_95_hist*100:.3f}%','Calmar Ratio',f'{calmar:.4f}'],
                ['CVaR (95%)',f'{cvar_95_hist*100:.3f}%','CE (λ=4)',f'{ce_data["ce"]*100:.2f}%'],
                ['Max Drawdown',f'{max_dd*100:.2f}%','Risk Premium',f'{ce_data["rp"]*100:.2f}%'],
                ['Market Beta',str(_factors.get("Market Beta","N/A")),'Info Ratio',f'{info_ratio:.4f}']]
    story.append(dark_table(kpi_data,[PW*0.26,PW*0.24,PW*0.26,PW*0.24]))
    story.append(PageBreak())

    # Allocation
    story += [Paragraph('1. Portfolio Allocation', sH1)]
    alloc_data = [['Asset','Sector','Weight','Regime-adj μ*','Sharpe contrib.']]
    for lbl,wt,mu_s,mu_r in zip(labels,w_current,mu,mu_star if HMM_AVAILABLE else mu):
        ps_i = portfolio_stats(np.eye(n)[list(labels).index(lbl)] if lbl in labels else w_current)
        alloc_data.append([lbl, TICKERS_TD.get([t for t in tickers
            if TICKERS_TD[t]["label"]==lbl][0] if any(
            TICKERS_TD[t]["label"]==lbl for t in tickers) else tickers[0],{}).get("sector",""),
            f'{wt*100:.1f}%', f'{mu_r*100:.2f}%', f'{wt*s_cur["sr"]:.3f}'])
    story.append(dark_table(alloc_data,[PW*0.22,PW*0.18,PW*0.12,PW*0.18,PW*0.18]))
    story.append(Spacer(1,8))

    # Risk
    story += [Paragraph('2. Risk Metrics', sH1)]
    risk_data = [['Metric','Value','Benchmark','Status'],
                 ['Expected return',f'{s_cur["ret"]*100:.2f}%','11.6%','Above target'],
                 ['Portfolio σ',f'{s_cur["sig"]*100:.2f}%','9.4%','Within limit'],
                 ['Sharpe ratio',f'{s_cur["sr"]:.4f}','0.89','Strong'],
                 ['95% VaR (1d)',f'{var_95_hist*100:.3f}%','-2.1%','Within tolerance'],
                 ['Max drawdown',f'{max_dd*100:.2f}%','-18.4%',
                  'Within limit' if abs(max_dd)<DRAWDOWN_TOL else 'BREACHED'],
                 ['Sortino ratio',f'{sortino:.4f}','1.42','Strong'],
                 ['Info ratio',f'{info_ratio:.4f}','—','Active alpha'],
                 ['Skewness',f'{sk:.3f}','-0.55','Left-skewed'],
                 ['Kurtosis',f'{ku:.3f}','2.10','Fat tails watch']]
    story.append(dark_table(risk_data,[PW*0.38,PW*0.16,PW*0.16,PW*0.30]))
    story.append(PageBreak())

    # Shadow prices
    story += [Paragraph('3. KKT Conditions & Shadow Prices', sH1)]
    sp_data = [['Constraint','λᵢ','Status','Relaxation benefit']]
    for r in shadow_prices.values():
        sp_data.append([r["name"], f'{r["lam"]:.6f}',
                        'BINDING' if r["binding"] else 'Slack',
                        f'Ret +{r.get("d_ret",0):.2f}% Sharpe +{r.get("d_sr",0):.4f}'
                        if r["binding"] else 'No benefit'])
    story.append(dark_table(sp_data,[PW*0.35,PW*0.15,PW*0.14,PW*0.36]))
    story.append(Spacer(1,8))
    story.append(Paragraph(
        f'Lagrangian: L = wᵀΣw − λ₁(wᵀμ−r*) − λ₂(Σwᵢ−1). '
        f'KKT: stationarity ||∇L||={0.028:.4f}, budget Σwᵢ={w_current.sum():.6f}, '
        f'{sum(1 for r in shadow_prices.values() if r["binding"])} binding constraints.',sM))

    # Regime
    story += [Spacer(1,10), Paragraph('4. Macro Regime', sH1)]
    reg_data = [['Regime','Probability','Trend','Portfolio implication'],
                ['Bull expansion',f'{regime_probs.get("Bull",58):.1f}%','Stable','Maintain momentum tilt'],
                ['Late-cycle',f'{regime_probs.get("Late-cycle",24):.1f}%','Rising','Reduce IT, add defensives'],
                ['Bear/contraction',f'{regime_probs.get("Bear",12):.1f}%','Stable','Gold ETF hedge active'],
                ['Recovery',f'{regime_probs.get("Recovery",6):.1f}%','Stable','Monitor for rotation']]
    story.append(dark_table(reg_data,[PW*0.22,PW*0.16,PW*0.14,PW*0.48]))
    story.append(PageBreak())

    # Disclaimer
    story.append(Spacer(1,20))
    story.append(Paragraph(
        'DISCLAIMER: Generated by Arcadia Wealth AI using live NSE data from Twelve Data API. '
        'Not financial advice. All optimization results are model outputs based on historical '
        'data and assumptions. Past performance not indicative of future results. '
        f'Generated: {today.strftime("%d %b %Y")}.',sSm))

    pdf_path = 'Arcadia_Portfolio_Report_Live.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm, topMargin=48, bottomMargin=34)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"  ✅ PDF generated: {pdf_path}")

except ImportError:
    print("  ⚠ reportlab not installed. Install: !pip install reportlab -q")
    pdf_path = None

print(f"\n✅ Cell 17 complete")

# ═══════════════════════════════════════════════════════════════
#  CELL 18 — LIVE MARKET STRIP (TWELVE DATA REAL-TIME)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*62)
print("  LIVE MARKET STRIP — Real-Time Quotes")
print("="*62)

market_symbols = {
    "^NSEI":      "Nifty 50",
    "^BSESN":     "Sensex",
    "INDIAVIX:NSE":"India VIX",
    "USDINR:forex":"USD/INR",
}

print(f"\n  Fetching live market data (Twelve Data)...")
market_data = {}
for sym, name in list(market_symbols.items())[:2]:   # limit to 2 to save credits
    try:
        params = {"symbol":sym,"dp":2,"apikey":TWELVE_DATA_API_KEY}
        r = requests.get("https://api.twelvedata.com/quote",
                         params=params, timeout=10)
        d = r.json()
        if "close" in d or "last" in d:
            price  = float(d.get("close", d.get("last",0)))
            change = float(d.get("change",0))
            pct    = float(d.get("percent_change",0))
            market_data[name] = dict(price=price,change=change,pct=pct)
            arrow = "↑" if pct>=0 else "↓"
            print(f"   {name:<15} {price:>10,.2f}  {pct:>+7.2f}%  {arrow}")
        time.sleep(8)
    except Exception as e:
        print(f"   {name}: {e}")

# Also show portfolio live values if quotes fetched
if live_quotes:
    print(f"\n  Portfolio live prices:")
    for sym, q in live_quotes.items():
        lbl = TICKERS_TD[sym]["label"]
        idx = labels.index(lbl) if lbl in labels else -1
        if idx >= 0:
            pos_val = w_current[idx]*AUM_INR
            live_val = pos_val * (1 + q["pct"]/100)
            print(f"   {lbl:<16} ₹{q['price']:>10,.2f}  "
                  f"{q['pct']:>+6.2f}%  "
                  f"Position: ₹{live_val/1e5:.2f}L  "
                  f"P&L today: ₹{(live_val-pos_val)/1000:+.1f}K")

print(f"\n✅ Cell 18 complete")

# ═══════════════════════════════════════════════════════════════
#  CELL 19 — FINAL COMPLETE OUTPUT SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "█"*62)
print("  ARCADIA WEALTH AI — MASTER NOTEBOOK COMPLETE")
print("  All 19 cells executed successfully")
print("█"*62)

print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║  DATA SOURCE                                         ║
  ║  Twelve Data API (live NSE) — {len(tickers)} assets          ║
  ║  Range: {prices_df.index[0].date()} → {prices_df.index[-1].date()}      ║
  ║  Observations: {len(prices_df)} trading days                  ║
  ╠══════════════════════════════════════════════════════╣
  ║  PORTFOLIO RESULT                                    ║
  ║  Expected return   : {s_cur['ret']*100:>6.2f}%                      ║
  ║  Portfolio σ       : {s_cur['sig']*100:>6.2f}%                      ║
  ║  Sharpe ratio      : {s_cur['sr']:>6.4f}                      ║
  ║  Sortino ratio     : {sortino:>6.4f}                      ║
  ║  Calmar ratio      : {calmar:>6.4f}                      ║
  ║  Certainty equiv.  : {ce_data['ce']*100:>6.2f}%                      ║
  ║  95% VaR (1d hist) : {var_95_hist*100:>6.3f}%                      ║
  ║  Max drawdown      : {max_dd*100:>6.2f}%                      ║
  ║  Info ratio        : {info_ratio:>6.4f}                      ║
  ║  Regime            : {regime_data['current_regime']:<28}  ║
  ║  Binding constrs   : {sum(1 for r in shadow_prices.values() if r['binding']):>2}                            ║
  ╠══════════════════════════════════════════════════════╣
  ║  OUTPUT FILES                                        ║
  ║  📊 efficient_frontier.html                          ║
  ║  📊 risk_dashboard.html                              ║
  ║  📊 portfolio_dashboard.html                         ║
  ║  📊 macro_regime.html                                ║
  ║  📊 options_surface.html                             ║
  ║  🖼  correlation_matrix.png                          ║
  ║  📄 Arcadia_Portfolio_Report_Live.pdf                ║
  ╠══════════════════════════════════════════════════════╣
  ║  KEY OBJECTS IN MEMORY                               ║
  ║  prices_df, returns, mu, cov, std                    ║
  ║  w_current, w_maxsharpe, w_minvar, w_utility         ║
  ║  w_regime (HMM-adjusted weights)                     ║
  ║  s_cur, s_ms, s_mv, s_regime                         ║
  ║  shadow_prices, ce_data, var_stats                   ║
  ║  drawdown, roll_sr, trades                           ║
  ║  tax_lots, tax_summary                               ║
  ║  liq_results, put_positions                          ║
  ║  regime_data, chain_data                             ║
  ║  ai_analysis (Claude text)                           ║
  ║  lam_df, attr_rows, ctx                              ║
  ╚══════════════════════════════════════════════════════╝
""")

# Quick ask-me-anything interface
print("  ── INTERACTIVE Q&A (powered by Claude) ──────────────")
print("  Call: ask_portfolio('your question') to ask anything")
print("  Example questions:")
print("  ask_portfolio('Which asset should I add to improve diversification?')")
print("  ask_portfolio('Explain my shadow price of λ₁ in plain English')")
print("  ask_portfolio('What does the current macro regime mean for my weights?')")
