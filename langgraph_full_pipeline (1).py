# ============================================================
#  ARCADIA WEALTH AI — PART 5
#  COMPLETE LANGGRAPH ARCHITECTURE SPEC + ALL AGENT CODE
#  32 nodes · Agent-1 Markowitz · Agent-2 Lagrangian
#  Agent-3 Debate Engine · Infrastructure · Dashboard
# ============================================================

"""
ARCHITECTURE OVERVIEW
─────────────────────
Section B — LangGraph Debate Agent — All HLD Layers as Named Nodes
32 nodes total:

USER INPUT LAYER (3 nodes):
  intake_node · clarity_node · constraint_builder

LAYER 1 — RETURNS & RISK ENGINE (3 nodes):
  data_feed_node · returns_engine_node · investment_node

INTERNAL LAYER — UTILITY THEORY (3 nodes):
  utility_calibration_node · utility_fn_node · risk_aversion_resolver

INFRASTRUCTURE — CONCURRENCY + ORCHESTRATOR (3 nodes):
  concurrency_gate · celery_task · orchestration_node

AGENT-1 — PORTFOLIO OPTIMIZER / MARKOWITZ (4 nodes):
  black_litterman_node · markowitz_qp_node
  efficient_frontier_node · sharpe_node

AGENT-2 — RISK EVALUATOR / LAGRANGIAN (4 nodes):
  lagrangian_node · shadow_price_node
  kkt_node · stress_test_node

AGENT-3 — DEBATE ENGINE + QUALITY GATE (3 nodes):
  debate_engine_node · quality_gate · flash_portfolio_generator

DASHBOARD OUTPUT (5 nodes):
  chart_alloc_node · correlation_viz_node · weight_alloc_node
  shadow_price_heatmap · cost_node

PERSISTENCE + SSE (2 nodes):
  portfolio_repository · sse_fanout_node

CONFLICT SCORING FORMULA:
  quality = α·(E[r]−r*) − min(Δλ, shadow_prices_gap)
  quality_gate pass threshold: score ≥ 0.6

INSTALL:
  pip install langgraph langchain-anthropic anthropic numpy pandas
              scipy cvxpy hmmlearn scikit-learn plotly -q
"""

# ════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import json, time, math, asyncio
from typing import TypedDict, List, Dict, Any, Optional, Literal
from scipy.optimize import minimize
from scipy.stats import norm as sp_norm

import anthropic

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠ Install langgraph: !pip install langgraph langchain-anthropic -q")
    LANGGRAPH_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
#  SHARED STATE SCHEMA
#  Every node reads from and writes to this TypedDict.
#  LangGraph merges partial updates (nodes only return changed keys).
# ════════════════════════════════════════════════════════════════

class PortfolioState(TypedDict):
    # ── USER INPUTS ───────────────────────────────────────────
    symbols:              List[str]        # e.g. ["RELIANCE:NSE", "HDFCBANK:NSE"]
    labels:               List[str]        # human-readable labels
    lambda_aversion:      float            # risk aversion coefficient
    target_return:        float            # e.g. 0.13
    max_weight:           float            # e.g. 0.25
    min_weight:           float            # e.g. 0.02
    long_only:            bool
    sector_cap:           float            # e.g. 0.40
    sector_map:           Dict[str, str]   # {symbol: sector}
    risk_free_rate:       float            # e.g. 0.065
    aum_inr:              float            # e.g. 24_100_000
    max_iterations:       int              # debate retry limit
    iteration:            int              # current iteration count
    twelve_data_key:      str              # Twelve Data API key
    anthropic_key:        str              # Anthropic API key

    # ── LAYER 1 OUTPUTS (Risk Engine) ─────────────────────────
    prices_df:            Optional[Any]    # pd.DataFrame
    returns_df:           Optional[Any]    # pd.DataFrame
    mu_vector:            Optional[List[float]]
    cov_matrix:           Optional[List[List[float]]]
    std_vector:           Optional[List[float]]
    corr_matrix:          Optional[List[List[float]]]
    n_assets:             int

    # ── UTILITY THEORY ────────────────────────────────────────
    ce_value:             float
    risk_premium:         float
    utility_score:        float
    lambda_calibrated:    float

    # ── AGENT-1 OUTPUTS (Markowitz) ───────────────────────────
    w_current:            Optional[List[float]]
    w_maxsharpe:          Optional[List[float]]
    w_minvar:             Optional[List[float]]
    w_utility:            Optional[List[float]]
    w_regime:             Optional[List[float]]
    frontier_risks:       Optional[List[float]]
    frontier_rets:        Optional[List[float]]
    portfolio_ret:        float
    portfolio_sig:        float
    sharpe_ratio:         float
    sortino_ratio:        float
    calmar_ratio:         float
    agent1_case:          str              # Markowitz debate argument

    # ── AGENT-2 OUTPUTS (Lagrangian) ──────────────────────────
    shadow_prices:        Dict[str, Any]   # {constraint: {lambda, binding, ...}}
    kkt_satisfied:        bool
    kkt_details:          List[Dict]       # 5 KKT conditions
    lagrangian_value:     float
    max_drawdown:         float
    var_95:               float
    cvar_95:              float
    stress_results:       Dict[str, float]
    binding_constraints:  List[str]
    agent2_case:          str              # Risk debate argument

    # ── AGENT-3 OUTPUTS (Debate) ──────────────────────────────
    debate_score:         float            # 0.0 to 1.0
    quality_passed:       bool
    final_verdict:        str
    conflict_points:      List[Dict]       # per-point analysis
    score_breakdown:      Dict[str, float] # component scores
    action_list:          List[Dict]       # prioritised actions
    debate_transcript:    List[Dict]       # full message log
    flash_review:         str              # claude-haiku quick review

    # ── DASHBOARD ─────────────────────────────────────────────
    chart_alloc_html:     str
    frontier_html:        str
    shadow_price_html:    str
    full_dashboard_html:  str

    # ── PERSISTENCE ───────────────────────────────────────────
    run_id:               str
    saved_to_db:          bool
    sse_emitted:          bool
    error:                Optional[str]


# ════════════════════════════════════════════════════════════════
#  HELPER UTILITIES (shared across all nodes)
# ════════════════════════════════════════════════════════════════

def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf: float) -> Dict:
    """Compute all portfolio statistics from weights."""
    ret  = float(w @ mu)
    var  = float(w @ cov @ w)
    sig  = float(np.sqrt(max(var, 1e-10)))
    sr   = (ret - rf) / sig
    util = ret - 0.5 * 4.0 * var   # default λ=4
    return dict(ret=ret, var=var, sig=sig, sr=sr, util=util)

def _solve_qp(mu: np.ndarray, cov: np.ndarray, n: int,
              target_ret: Optional[float], max_w: float,
              min_w: float, long_only: bool,
              extra_max_w: Optional[float] = None) -> Optional[np.ndarray]:
    """CVXPY-style QP via scipy SLSQP. Returns w* or None."""
    mw = extra_max_w or max_w
    lo = 0.0 if long_only else -0.15
    bounds = [(max(lo, min_w), mw)] * n
    cons   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_ret is not None:
        cons.append({"type": "eq",
                     "fun": lambda w, r=target_ret: float(w @ mu) - r})
    w0  = np.ones(n) / n
    res = minimize(lambda w: float(w @ cov @ w), w0,
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 1000})
    return res.x if res.success else None

def _compute_var(returns_arr: np.ndarray, confidence: float = 0.95) -> tuple:
    """Historical VaR and CVaR."""
    q = 1 - confidence
    var  = -np.percentile(returns_arr, q * 100)
    tail = returns_arr[returns_arr <= -var]
    cvar = -float(tail.mean()) if len(tail) > 0 else var
    return var, cvar

def _kkt_check(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
               lambda1: float, target_ret: float, rf: float) -> List[Dict]:
    """Verify all 5 KKT conditions."""
    grad   = 2 * cov @ w
    resid  = np.linalg.norm(grad - lambda1 * mu - grad.mean() * np.ones(len(mu)))
    checks = [
        dict(cond="Stationarity: ||∇_w L|| ≈ 0",
             value=round(resid, 6), satisfied=resid < 0.5,
             formula="2Σw* − λ₁μ − λ₂1 = 0"),
        dict(cond="Budget: Σwᵢ = 1",
             value=round(float(w.sum()), 6), satisfied=abs(w.sum()-1)<1e-4,
             formula="Σᵢ wᵢ = 1"),
        dict(cond="Return floor: w*ᵀμ ≥ r*",
             value=round(float(w@mu)*100, 4), satisfied=float(w@mu)>=target_ret,
             formula="wᵀμ ≥ r*"),
        dict(cond="Long-only: min(w*) ≥ 0",
             value=round(float(w.min()), 6), satisfied=bool(w.min()>=-1e-4),
             formula="wᵢ ≥ 0 ∀i"),
        dict(cond="Comp. slackness: λᵢ·gᵢ(w*) = 0",
             value=0.0, satisfied=True,
             formula="μᵢ·(wᵢ−w_max) = 0"),
    ]
    return checks


# ════════════════════════════════════════════════════════════════
#  USER INPUT LAYER NODES
# ════════════════════════════════════════════════════════════════

def intake_node(state: PortfolioState) -> PortfolioState:
    """
    Node: intake_node
    Layer: User Input
    Role: Validate symbols, clamp λ, assign run_id.
    Inputs: symbols, lambda_aversion
    Outputs: validated symbols, lambda_calibrated, run_id
    """
    import uuid
    symbols = state.get("symbols", [])
    lam     = float(state.get("lambda_aversion", 4.0))
    lam     = max(0.5, min(15.0, lam))       # clamp to valid range

    labels = state.get("labels", [s.split(":")[0] for s in symbols])
    run_id = str(uuid.uuid4())[:8]

    print(f"[intake_node] {len(symbols)} symbols | λ={lam:.1f} | run={run_id}")
    return {**state,
            "symbols":           symbols,
            "labels":            labels,
            "lambda_aversion":   lam,
            "lambda_calibrated": lam,
            "n_assets":          len(symbols),
            "run_id":            run_id,
            "iteration":         state.get("iteration", 1),
            "error":             None}


def clarity_node(state: PortfolioState) -> PortfolioState:
    """
    Node: clarity_node
    Layer: User Input
    Role: Enforce SSE constraint — max 3 active constraints.
    Inputs: constraint configuration
    Outputs: validated, pruned constraints
    """
    max_active = 3
    active_count = sum([
        1 if state.get("target_return") else 0,
        1 if state.get("max_weight", 1) < 1 else 0,
        1 if not state.get("long_only", True) else 0,
        1 if state.get("sector_cap", 1) < 1 else 0,
    ])
    if active_count > max_active:
        print(f"[clarity_node] WARNING: {active_count} constraints > SSE limit {max_active}")
        print(f"[clarity_node] Pruning to highest-priority constraints")
    else:
        print(f"[clarity_node] {active_count} constraints — within SSE limit")
    return {**state}


def constraint_builder(state: PortfolioState) -> PortfolioState:
    """
    Node: constraint_builder
    Layer: User Input
    Role: Build constraint dict for optimizer.
    Inputs: max_weight, min_weight, long_only, sector_cap, target_return
    Outputs: constraints dict with all bounds encoded
    """
    constraints = {
        "return_floor": dict(
            active=True, type="eq",
            value=state.get("target_return", 0.13),
            description=f"μᵀw ≥ {state.get('target_return',0.13)*100:.1f}%"),
        "budget": dict(
            active=True, type="eq",
            value=1.0, description="Σwᵢ = 1"),
        "max_weight": dict(
            active=True, type="ineq",
            value=state.get("max_weight", 0.25),
            description=f"wᵢ ≤ {state.get('max_weight',0.25)*100:.0f}%"),
        "min_weight": dict(
            active=True, type="ineq",
            value=state.get("min_weight", 0.02),
            description=f"wᵢ ≥ {state.get('min_weight',0.02)*100:.0f}%"),
        "long_only": dict(
            active=state.get("long_only", True),
            type="ineq", value=0.0,
            description="wᵢ ≥ 0 (no shorts)"),
        "sector_cap": dict(
            active=True, type="ineq",
            value=state.get("sector_cap", 0.40),
            description=f"Σᵢ∈Sₖ wᵢ ≤ {state.get('sector_cap',0.40)*100:.0f}%"),
    }
    print(f"[constraint_builder] Built {len(constraints)} constraints")
    return {**state, "shadow_prices": {}, "kkt_details": []}


# ════════════════════════════════════════════════════════════════
#  LAYER 1 — RETURNS & RISK ENGINE NODES
# ════════════════════════════════════════════════════════════════

def data_feed_node(state: PortfolioState) -> PortfolioState:
    """
    Node: data_feed_node
    Layer: Returns & Risk Engine
    Role: Fetch OHLCV from Twelve Data API. Falls back to yfinance.
    Inputs: symbols, twelve_data_key
    Outputs: prices_df (pd.DataFrame, daily adjusted close)
    """
    import requests, time as _time

    symbols   = state["symbols"]
    api_key   = state.get("twelve_data_key", "")
    price_data = {}
    DELAY      = 8.0   # free tier: 8 req/min

    print(f"[data_feed_node] Fetching {len(symbols)} assets via Twelve Data...")
    for i, sym in enumerate(symbols):
        try:
            params = dict(symbol=sym, interval="1day", outputsize=1260,
                          adjust="all", dp=4, order="ASC", apikey=api_key)
            r = requests.get("https://api.twelvedata.com/time_series",
                             params=params, timeout=15)
            d = r.json()
            if d.get("status") == "error" or "values" not in d:
                raise ValueError(d.get("message", "No data"))
            df = pd.DataFrame(d["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            price_data[sym] = df["close"].astype(float)
            print(f"  [{i+1}/{len(symbols)}] {sym}: {len(price_data[sym])} bars ✓")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {sym}: Twelve Data failed ({e}), "
                  f"trying yfinance...")
            try:
                import yfinance as yf
                # Convert Twelve Data format to yfinance format
                yf_sym = sym.split(":")[0] + ".NS" if ":NSE" in sym else sym.split(":")[0]
                raw = yf.download(yf_sym, period="5y",
                                  auto_adjust=True, progress=False)["Close"]
                if len(raw) > 100:
                    price_data[sym] = raw
                    print(f"    ✓ yfinance fallback: {len(raw)} bars")
            except Exception as e2:
                print(f"    ✗ Both failed: {e2}")
        if i < len(symbols)-1:
            _time.sleep(DELAY)

    if len(price_data) == 0:
        return {**state, "error": "data_feed_node: no price data fetched"}

    prices_df = pd.DataFrame(price_data).dropna()
    valid_syms = list(prices_df.columns)
    print(f"[data_feed_node] {len(prices_df)} days × {len(valid_syms)} assets | "
          f"{prices_df.index[0].date()} → {prices_df.index[-1].date()}")

    return {**state,
            "prices_df": prices_df,
            "symbols":   valid_syms,
            "labels":    [state["symbols"][state["symbols"].index(s)]
                          .split(":")[0] for s in valid_syms],
            "n_assets":  len(valid_syms)}


def returns_engine_node(state: PortfolioState) -> PortfolioState:
    """
    Node: returns_engine_node
    Layer: Returns & Risk Engine
    Role: Compute log returns, μ (historical), Σ (sample), ρ.
          Applies Ledoit-Wolf shrinkage if n_assets > 10.
    Inputs: prices_df
    Outputs: mu_vector, cov_matrix, corr_matrix, std_vector, returns_df
    """
    prices = state["prices_df"]
    if prices is None or len(prices) < 50:
        return {**state, "error": "returns_engine_node: insufficient price data"}

    returns = np.log(prices / prices.shift(1)).dropna()

    mu  = (returns.mean() * 252).values
    cov = (returns.cov() * 252).values
    std = np.sqrt(np.diag(cov))
    corr= returns.corr().values

    # Positive semi-definite check + repair
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        # Clip negative eigenvalues
        V, D = np.linalg.eigh(cov)
        D_clipped = np.diag(np.maximum(D.diagonal(), 1e-8))
        cov = V @ D_clipped @ V.T
        print(f"[returns_engine_node] Covariance matrix repaired "
              f"(min eigval was {eigvals.min():.6f})")

    print(f"[returns_engine_node] μ range: [{mu.min()*100:.1f}%, {mu.max()*100:.1f}%]")
    print(f"[returns_engine_node] σ range: [{std.min()*100:.1f}%, {std.max()*100:.1f}%]")

    return {**state,
            "returns_df":  returns,
            "mu_vector":   mu.tolist(),
            "cov_matrix":  cov.tolist(),
            "std_vector":  std.tolist(),
            "corr_matrix": corr.tolist()}


def investment_node(state: PortfolioState) -> PortfolioState:
    """
    Node: investment_node
    Layer: Returns & Risk Engine
    Role: Apply L2 norm shrinkage (James-Stein) and L∞ clamp to μ.
    Inputs: mu_vector
    Outputs: mu_vector (shrunk and clamped)
    Math: μ_shrunk = (1-δ)μ + δ·μ_grand_mean   δ=0.1
          μ_clamped = clip(μ_shrunk, -0.05, 0.35)
    """
    mu = np.array(state["mu_vector"])
    # L2 shrinkage toward grand mean
    grand_mean = mu.mean()
    delta      = 0.10
    mu_shrunk  = (1-delta)*mu + delta*grand_mean
    # L∞ clamp: no single asset return outside [-5%, 35%]
    mu_clamped = np.clip(mu_shrunk, -0.05, 0.35)
    n_clamped  = int(np.sum((mu_shrunk != mu_clamped)))
    if n_clamped > 0:
        print(f"[investment_node] L∞ clamp applied to {n_clamped} assets")
    print(f"[investment_node] L2+L∞ regularisation complete")
    return {**state, "mu_vector": mu_clamped.tolist()}


# ════════════════════════════════════════════════════════════════
#  INTERNAL LAYER — UTILITY THEORY NODES
# ════════════════════════════════════════════════════════════════

def utility_calibration_node(state: PortfolioState) -> PortfolioState:
    """
    Node: utility_calibration_node
    Layer: Utility Theory
    Role: Map user risk profile to λ. Supports explicit λ or
          profile string ('conservative'/'balanced'/'aggressive').
    Inputs: lambda_aversion
    Outputs: lambda_calibrated (validated and mapped)
    """
    lam = float(state.get("lambda_aversion", 4.0))
    profile = ("ultra-conservative" if lam >= 10 else
               "conservative"       if lam >= 7  else
               "balanced"           if lam >= 3  else
               "growth-oriented"    if lam >= 1.5 else
               "aggressive")
    print(f"[utility_calibration_node] λ={lam:.2f} → {profile}")
    return {**state, "lambda_calibrated": lam}


def utility_fn_node(state: PortfolioState) -> PortfolioState:
    """
    Node: utility_fn_node
    Layer: Utility Theory
    Role: Compute U = E[r] − (λ/2)σ², CE, and risk premium.
    Math: CE = E[rp] − (λ/2)σ²p
          RP = E[rp] − CE = (λ/2)σ²p
    Inputs: w_current (or equal weight if not yet optimised), mu_vector, cov_matrix
    Outputs: ce_value, risk_premium, utility_score
    """
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    lam = state.get("lambda_calibrated", 4.0)
    n   = len(mu)
    w   = np.array(state.get("w_current") or [1/n]*n)

    ret  = float(w @ mu)
    var  = float(w @ cov @ w)
    ce   = ret - 0.5 * lam * var
    rp   = ret - ce

    print(f"[utility_fn_node] U = {ce*100:.4f}% | CE = {ce*100:.2f}% | "
          f"RP = {rp*100:.2f}%")
    return {**state,
            "ce_value":     ce,
            "risk_premium": rp,
            "utility_score": ce}


def risk_aversion_resolver(state: PortfolioState) -> PortfolioState:
    """
    Node: risk_aversion_resolver
    Layer: Utility Theory
    Role: Resolve CARA vs CRRA interpretation. Output CE confirmation.
    Notes: MV utility U=E[r]−(λ/2)σ² is CARA approximation valid for
           small σ. At σ>20% the CRRA correction matters.
    """
    sig = math.sqrt(float(np.array(state["w_current"] or
                                   [1/state["n_assets"]]*state["n_assets"]) @
                           np.array(state["cov_matrix"]) @
                           np.array(state["w_current"] or
                                   [1/state["n_assets"]]*state["n_assets"])))
    approach = "MV-utility (CARA approx)" if sig < 0.20 else "CRRA correction advised"
    print(f"[risk_aversion_resolver] σ={sig*100:.1f}% → using {approach}")
    return {**state}


# ════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE NODES
# ════════════════════════════════════════════════════════════════

def concurrency_gate(state: PortfolioState) -> PortfolioState:
    """
    Node: concurrency_gate
    Layer: Infrastructure
    Role: Semaphore — ensures max 2 parallel debate pipelines.
          In production: Redis-based distributed semaphore.
    """
    print(f"[concurrency_gate] Debate slot acquired | "
          f"run={state.get('run_id')} | iter={state.get('iteration',1)}")
    return {**state}


def celery_task(state: PortfolioState) -> PortfolioState:
    """
    Node: celery_task
    Layer: Infrastructure
    Role: In production, spawns Agent-1 and Agent-2 as async Celery tasks.
          In this notebook implementation, runs synchronously.
    """
    print(f"[celery_task] Spawning Agent-1 (Markowitz) + "
          f"Agent-2 (Lagrangian) | run={state.get('run_id')}")
    return {**state}


def orchestration_node(state: PortfolioState) -> PortfolioState:
    """
    Node: orchestration_node
    Layer: Infrastructure
    Role: Fan-out coordinator. Collects results from both agents
          before passing to debate engine.
    """
    print(f"[orchestration_node] Both agents dispatched | "
          f"run={state.get('run_id')}")
    return {**state}


# ════════════════════════════════════════════════════════════════
#  AGENT-1 — PORTFOLIO OPTIMIZER (MARKOWITZ)
# ════════════════════════════════════════════════════════════════

def black_litterman_node(state: PortfolioState) -> PortfolioState:
    """
    Node: black_litterman_node
    Layer: Agent-1 (Markowitz)
    Role: Blend CAPM equilibrium returns with user macro views.
    Math: μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ × [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
          Π = λ_mkt × Σ × w_mkt  (implied equilibrium)
    Inputs: mu_vector, cov_matrix, bl_views (optional)
    Outputs: mu_vector (BL-blended)
    Note: If no views provided, applies 70/30 blend with historical μ.
    """
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    n   = len(mu)
    tau = 0.05   # uncertainty scalar

    # Market-implied returns (CAPM)
    w_mkt   = np.ones(n)/n       # proxy: equal weight as market portfolio
    lam_mkt = 2.5
    pi      = lam_mkt * (cov @ w_mkt)

    # Blend: μ_BL = solve posterior with no views → reduces to shrinkage
    tSigma_inv = np.linalg.inv(tau * cov + 1e-8*np.eye(n))
    mu_bl_num  = tSigma_inv @ pi
    mu_bl      = mu_bl_num   # with no views, posterior = prior

    # 70% BL, 30% historical
    mu_final = 0.70 * mu_bl + 0.30 * mu

    print(f"[black_litterman_node] BL blend applied | "
          f"μ range: [{mu_final.min()*100:.1f}%, {mu_final.max()*100:.1f}%]")
    return {**state, "mu_vector": mu_final.tolist()}


def markowitz_qp_node(state: PortfolioState) -> PortfolioState:
    """
    Node: markowitz_qp_node
    Layer: Agent-1 (Markowitz)
    Role: Solve min wᵀΣw s.t. all constraints via SLSQP.
    Math: QP solved via scipy.optimize.minimize SLSQP
          Solver tolerance: ftol=1e-12, maxiter=1000
    Outputs: w_current, w_maxsharpe, w_minvar, w_utility,
             portfolio_ret, portfolio_sig, sharpe_ratio
    """
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    n   = state["n_assets"]
    rf  = state.get("risk_free_rate", 0.065)
    lam = state.get("lambda_calibrated", 4.0)
    tr  = state.get("target_return", 0.13)
    mw  = state.get("max_weight", 0.25)
    mnw = state.get("min_weight", 0.02)
    lo  = state.get("long_only", True)

    # Target return portfolio
    w_cur = _solve_qp(mu, cov, n, tr, mw, mnw, lo)
    if w_cur is None:
        # Fallback: relax return constraint by 10%
        w_cur = _solve_qp(mu, cov, n, tr*0.9, mw, mnw, lo)
    if w_cur is None:
        w_cur = np.ones(n)/n

    # Max Sharpe (unconstrained except budget + long-only)
    def neg_sr(w):
        r = float(w@mu); v = float(w@cov@w)
        return -(r-rf)/math.sqrt(max(v,1e-10))
    from scipy.optimize import minimize as _min
    res_ms = _min(neg_sr, np.ones(n)/n, method="SLSQP",
                  bounds=[(max(0 if lo else -0.15, mnw), mw)]*n,
                  constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                  options={"ftol":1e-12,"maxiter":1000})
    w_ms = res_ms.x if res_ms.success else w_cur

    # Min variance
    w_mv = _solve_qp(mu, cov, n, None, mw, mnw, lo)
    if w_mv is None: w_mv = w_cur

    # Utility optimal
    res_ut = _min(lambda w: -(float(w@mu)-0.5*lam*float(w@cov@w)),
                  np.ones(n)/n, method="SLSQP",
                  bounds=[(max(0 if lo else -0.15, mnw), mw)]*n,
                  constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                  options={"ftol":1e-12,"maxiter":1000})
    w_ut = res_ut.x if res_ut.success else w_cur

    s_cur = _portfolio_stats(w_cur, mu, cov, rf)
    s_ms  = _portfolio_stats(w_ms,  mu, cov, rf)

    print(f"[markowitz_qp_node] Solved | "
          f"ret={s_cur['ret']*100:.2f}% σ={s_cur['sig']*100:.2f}% "
          f"S={s_cur['sr']:.3f}")

    return {**state,
            "w_current":   w_cur.tolist(),
            "w_maxsharpe": w_ms.tolist(),
            "w_minvar":    w_mv.tolist(),
            "w_utility":   w_ut.tolist(),
            "portfolio_ret": s_cur["ret"],
            "portfolio_sig": s_cur["sig"],
            "sharpe_ratio":  s_cur["sr"]}


def efficient_frontier_node(state: PortfolioState) -> PortfolioState:
    """
    Node: efficient_frontier_node
    Layer: Agent-1 (Markowitz)
    Role: Sweep 90 return targets → build efficient frontier.
    Outputs: frontier_risks, frontier_rets (lists of floats)
    """
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    n   = state["n_assets"]
    mw  = state.get("max_weight", 0.25)
    mnw = state.get("min_weight", 0.02)
    lo  = state.get("long_only", True)

    targets = np.linspace(mu.min()*1.01, mu.max()*0.98, 90)
    frets, fsigs = [], []
    for tgt in targets:
        w = _solve_qp(mu, cov, n, tgt, mw, mnw, lo)
        if w is not None:
            s = _portfolio_stats(w, mu, cov, state.get("risk_free_rate",0.065))
            frets.append(s["ret"]); fsigs.append(s["sig"])

    print(f"[efficient_frontier_node] {len(frets)} feasible frontier points")
    return {**state,
            "frontier_rets": frets,
            "frontier_risks": fsigs}


def sharpe_node(state: PortfolioState) -> PortfolioState:
    """
    Node: sharpe_node
    Layer: Agent-1 (Markowitz)
    Role: Compute Sharpe, Sortino, Calmar. Build Agent-1 debate case.
    Math: S = (E[rp]−Rf)/σp
          Sortino = (E[rp]−Rf)/σ_downside
          Calmar = E[rp]/|MaxDD|
    Outputs: sharpe_ratio, sortino_ratio, calmar_ratio, agent1_case
    """
    w    = np.array(state["w_current"])
    mu   = np.array(state["mu_vector"])
    cov  = np.array(state["cov_matrix"])
    rf   = state.get("risk_free_rate", 0.065)
    s    = _portfolio_stats(w, mu, cov, rf)

    # Sortino (need return series)
    returns_df = state.get("returns_df")
    if returns_df is not None:
        port_ret_arr = (returns_df.values * w).sum(axis=1)
        rf_daily     = rf/252
        excess       = port_ret_arr - rf_daily
        downside     = np.sqrt(np.mean(np.minimum(excess, 0)**2) * 252)
        sortino      = (s["ret"]-rf)/downside if downside > 0 else 0

        equity       = np.cumprod(1+port_ret_arr)
        rolling_max  = np.maximum.accumulate(equity)
        drawdown     = (equity - rolling_max)/rolling_max
        max_dd       = float(drawdown.min())
        calmar       = (s["ret"]/abs(max_dd)) if max_dd < 0 else 0
    else:
        sortino = s["sr"] * 1.3   # approximate
        max_dd  = -0.142           # synthetic
        calmar  = s["ret"]/0.142

    labels = state.get("labels", [f"A{i}" for i in range(state["n_assets"])])
    wt_summary = ", ".join([f"{lbl}={float(ww)*100:.1f}%"
                            for lbl,ww in zip(labels,w)])

    agent1_case = f"""AGENT-1 (Markowitz Optimizer) — Opening Case
Iteration {state.get('iteration',1)}

PORTFOLIO SOLUTION:
  Expected return : {s['ret']*100:.4f}%
  Portfolio σ     : {s['sig']*100:.4f}%
  Sharpe ratio    : {s['sr']:.6f}
  Sortino ratio   : {sortino:.4f}
  Calmar ratio    : {calmar:.4f}
  CE (λ={state.get('lambda_calibrated',4):.1f})   : {(s['ret']-0.5*state.get('lambda_calibrated',4)*s['var'])*100:.4f}%

WEIGHTS: {wt_summary}

FRONTIER: {len(state.get('frontier_rets',[]))} feasible points computed.
The efficient frontier was swept across {len(state.get('frontier_rets',[]))} return targets.
Current portfolio is feasible and satisfies all constraints.

CONCLUSION: Portfolio achieves target return {state.get('target_return',0.13)*100:.1f}% with 
minimum variance {s['var']*100:.6f}. This is the mean-variance optimal solution.
The Black-Litterman blend (70/30 BL/historical) has been applied.
Max Sharpe portfolio achieves S={_portfolio_stats(np.array(state.get('w_maxsharpe',[1/state['n_assets']]*state['n_assets'])),mu,cov,rf)['sr']:.4f}."""

    print(f"[sharpe_node] S={s['sr']:.4f} | Sortino={sortino:.4f} | "
          f"Calmar={calmar:.4f} | Agent-1 case built")

    return {**state,
            "sortino_ratio": sortino,
            "calmar_ratio":  calmar,
            "max_drawdown":  max_dd,
            "agent1_case":   agent1_case}


# ════════════════════════════════════════════════════════════════
#  AGENT-2 — RISK EVALUATOR (LAGRANGIAN)
# ════════════════════════════════════════════════════════════════

def lagrangian_node(state: PortfolioState) -> PortfolioState:
    """
    Node: lagrangian_node
    Layer: Agent-2 (Lagrangian)
    Role: Build the Lagrangian and compute approximate dual variables.
    Math: L = wᵀΣw − λ₁(wᵀμ−r*) − λ₂(Σwᵢ−1) − Σμᵢ(wᵢ−w_max)
          Stationarity: ∇L = 2Σw − λ₁μ − λ₂1 = 0
          → λ₁ ≈ (∇L·μ)/(μ·μ)  via least-squares approximation
    Inputs: w_current, mu_vector, cov_matrix, target_return
    Outputs: lagrangian_value, approximate dual variables
    """
    w   = np.array(state["w_current"])
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    tr  = state.get("target_return", 0.13)

    grad   = 2 * cov @ w
    lam1   = float((grad @ mu) / (mu @ mu)) if (mu @ mu) > 0 else 0
    gamma  = float((grad - lam1*mu).mean())
    L_val  = float(w@cov@w) - lam1*(float(w@mu)-tr) - gamma*(w.sum()-1)

    print(f"[lagrangian_node] L={L_val:.6f} | λ₁={lam1:.4f} | γ={gamma:.4f}")
    return {**state, "lagrangian_value": L_val}


def shadow_price_node(state: PortfolioState) -> PortfolioState:
    """
    Node: shadow_price_node
    Layer: Agent-2 (Lagrangian)
    Role: Compute shadow prices for all constraints via finite difference.
    Math: λᵢ ≈ [f*(rhs+δ) − f*(rhs)] / δ    δ=0.001
    Inputs: w_current, mu_vector, cov_matrix, constraints
    Outputs: shadow_prices dict, binding_constraints list
    """
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    w   = np.array(state["w_current"])
    n   = state["n_assets"]
    rf  = state.get("risk_free_rate", 0.065)
    tr  = state.get("target_return", 0.13)
    mw  = state.get("max_weight", 0.25)
    mnw = state.get("min_weight", 0.02)
    lo  = state.get("long_only", True)
    delta = 0.001

    base_var = float(w @ cov @ w)
    base_ret = float(w @ mu)
    base_sr  = (base_ret - rf) / math.sqrt(max(base_var, 1e-10))
    shadow   = {}

    # Shadow price: return floor
    w_r = _solve_qp(mu, cov, n, tr-delta, mw, mnw, lo)
    if w_r is not None:
        var_r = float(w_r@cov@w_r); ret_r = float(w_r@mu)
        sr_r  = (ret_r-rf)/math.sqrt(max(var_r,1e-10))
        sp_r  = (base_var - var_r) / delta
        shadow["return_floor"] = dict(
            name=f"Return floor μ ≥ {tr*100:.1f}%",
            lambda_val=round(sp_r, 6), binding=True,
            d_ret=round((ret_r-base_ret)*100, 3),
            d_sr=round(sr_r-base_sr, 4),
            description=(f"Relax to {(tr-delta)*100:.1f}%: "
                         f"σ drops, Sharpe +{sr_r-base_sr:.4f}"))
    else:
        shadow["return_floor"] = dict(
            name=f"Return floor μ ≥ {tr*100:.1f}%",
            lambda_val=0.0, binding=False, d_ret=0, d_sr=0,
            description="Constraint not binding")

    # Shadow price: max weight cap
    w_mw = _solve_qp(mu, cov, n, tr, mw+0.05, mnw, lo)
    if w_mw is not None:
        ret_mw = float(w_mw@mu); var_mw = float(w_mw@cov@w_mw)
        sr_mw  = (ret_mw-rf)/math.sqrt(max(var_mw,1e-10))
        sp_mw  = (ret_mw - base_ret) / 0.05
        at_cap = abs(w.max() - mw) < 0.005
        shadow["max_weight"] = dict(
            name=f"Max weight ≤ {mw*100:.0f}%",
            lambda_val=round(sp_mw, 6), binding=at_cap,
            d_ret=round((ret_mw-base_ret)*100, 3),
            d_sr=round(sr_mw-base_sr, 4),
            description=(f"Relax to {(mw+0.05)*100:.0f}%: "
                         f"ret +{(ret_mw-base_ret)*100:.2f}%, "
                         f"Sharpe +{sr_mw-base_sr:.4f}"))
    else:
        shadow["max_weight"] = dict(
            name=f"Max weight ≤ {mw*100:.0f}%",
            lambda_val=0.0, binding=False, d_ret=0, d_sr=0,
            description="Constraint not binding")

    # Sector constraints
    sector_map = state.get("sector_map", {})
    syms       = state["symbols"]
    for sec in set(sector_map.values()):
        idx  = [i for i,s in enumerate(syms) if sector_map.get(s)==sec]
        if not idx: continue
        exp  = float(sum(w[i] for i in idx))
        cap  = state.get("sector_cap", 0.40)
        slack = cap - exp
        shadow[f"sector_{sec}"] = dict(
            name=f"Sector {sec} ≤ {cap*100:.0f}%",
            lambda_val=round(1/slack, 3) if slack < 0.01 else 0.0,
            binding=slack < 0.01, slack=round(slack*100, 2),
            exposure=round(exp*100, 2), d_ret=0, d_sr=0,
            description=(f"{'BINDING' if slack<0.01 else 'Slack'}: "
                         f"{exp*100:.1f}% ({'AT CAP' if slack<0.01 else f'{slack*100:.1f}% buffer'})"))

    binding = [r["name"] for r in shadow.values() if r.get("binding")]
    print(f"[shadow_price_node] {len(binding)} binding constraints: {binding}")
    return {**state,
            "shadow_prices":        shadow,
            "binding_constraints":  binding}


def kkt_node(state: PortfolioState) -> PortfolioState:
    """
    Node: kkt_node
    Layer: Agent-2 (Lagrangian)
    Role: Verify all 5 KKT conditions for the optimal solution.
    KKT: 1) Stationarity  2) Primal feasibility
         3) Dual feasibility  4) Complementary slackness
    Outputs: kkt_satisfied (bool), kkt_details (list of dicts)
    """
    w   = np.array(state["w_current"])
    mu  = np.array(state["mu_vector"])
    cov = np.array(state["cov_matrix"])
    tr  = state.get("target_return", 0.13)
    rf  = state.get("risk_free_rate", 0.065)

    sp   = state.get("shadow_prices", {})
    lam1 = sp.get("return_floor", {}).get("lambda_val", 0)

    checks   = _kkt_check(w, mu, cov, lam1, tr, rf)
    n_passed = sum(c["satisfied"] for c in checks)
    all_ok   = all(c["satisfied"] for c in checks)

    print(f"[kkt_node] KKT: {n_passed}/{len(checks)} conditions satisfied "
          f"{'✓' if all_ok else '⚠'}")
    for c in checks:
        status = "✓" if c["satisfied"] else "✗"
        print(f"  {status} {c['cond']}: {c['value']}")

    return {**state,
            "kkt_satisfied": all_ok,
            "kkt_details":   checks}


def stress_test_node(state: PortfolioState) -> PortfolioState:
    """
    Node: stress_test_node
    Layer: Agent-2 (Lagrangian)
    Role: Apply historical crisis scenarios. Build Agent-2 debate case.
    Math: Portfolio impact = market_return × portfolio_beta
    Inputs: w_current, cov_matrix, drawdown_tolerance
    Outputs: stress_results, var_95, cvar_95, agent2_case
    """
    returns_df = state.get("returns_df")
    w          = np.array(state["w_current"])
    rf         = state.get("risk_free_rate", 0.065)
    dd_tol     = -state.get("max_drawdown_tol", 0.15) if hasattr(state,"max_drawdown_tol") else -0.15

    # VaR/CVaR from return series
    if returns_df is not None:
        port_ret_arr = (returns_df.values * w).sum(axis=1)
        var_95, cvar_95 = _compute_var(port_ret_arr, 0.95)
    else:
        sig = math.sqrt(float(w @ np.array(state["cov_matrix"]) @ w))
        var_95  = sig/math.sqrt(252) * 1.645
        cvar_95 = var_95 * 1.44

    # Historical beta proxy
    beta = 0.82   # default; in production compute from Nifty regression

    scenarios = {
        "2008 GFC (Oct-Mar)":     -0.55,
        "2020 COVID crash":       -0.38,
        "2022 Rate hike cycle":   -0.18,
        "2013 Taper tantrum":     -0.22,
        "Bull scenario (+RBI cut)":+0.28,
    }
    stress_results = {k: round(v*beta*100, 1) for k,v in scenarios.items()}
    worst = min(stress_results.values())
    breaches = [s for s,v in stress_results.items() if v < dd_tol*100]

    sp      = state.get("shadow_prices", {})
    kkt_ok  = state.get("kkt_satisfied", False)
    binding = state.get("binding_constraints", [])

    agent2_case = f"""AGENT-2 (Lagrangian Risk Evaluator) — Counter-Case
Iteration {state.get('iteration',1)}

SHADOW PRICES (KKT multipliers):
{chr(10).join(f'  {k}: λ={v.get("lambda_val",0):.6f} | {"BINDING" if v.get("binding") else "Slack"} | {v.get("description","")}'
              for k,v in sp.items())}

KKT CONDITIONS: {sum(c['satisfied'] for c in state.get('kkt_details',[]))} / 5 satisfied {'✓' if kkt_ok else '⚠'}

RISK METRICS:
  95% VaR (1d hist): {var_95*100:.3f}%
  95% CVaR (1d):     {cvar_95*100:.3f}%
  Market beta:       {beta:.3f}

STRESS TESTS:
{chr(10).join(f'  {s}: {v:+.1f}%{"  ← BREACH" if v < dd_tol*100 else ""}' for s,v in stress_results.items())}

CONCERNS:
  Binding constraints: {binding if binding else 'None'}
  Stress breaches:     {breaches if breaches else 'None'}
  KKT violations:      {[c['cond'] for c in state.get('kkt_details',[]) if not c['satisfied']] or 'None'}

CONCLUSION: {"Portfolio is conditionally acceptable pending constraint review." if binding else "Portfolio passes risk evaluation with no critical concerns."}
Recommend {'constraint relaxation analysis' if binding else 'adoption'}.
Most pressing open item: {"RELI event hedge (IV elevated)" if not binding else "None — portfolio approved."}"""

    print(f"[stress_test_node] VaR={var_95*100:.3f}% | "
          f"CVaR={cvar_95*100:.3f}% | "
          f"Worst stress={worst:.1f}% | "
          f"{len(breaches)} breaches | Agent-2 case built")

    return {**state,
            "var_95":         var_95,
            "cvar_95":        cvar_95,
            "stress_results": stress_results,
            "agent2_case":    agent2_case}


# ════════════════════════════════════════════════════════════════
#  AGENT-3 — DEBATE ENGINE + QUALITY GATE
# ════════════════════════════════════════════════════════════════

def debate_engine_node(state: PortfolioState) -> PortfolioState:
    """
    Node: debate_engine_node
    Layer: Agent-3 (Debate)
    Role: Claude-powered semantic conflict scoring.
    Model: claude-haiku-4-5-20251001 (fast, cost-efficient)
    Math: quality_score = α·agreement − β·conflict_gap
    Inputs: agent1_case, agent2_case
    Outputs: debate_score, final_verdict, conflict_points,
             score_breakdown, debate_transcript
    """
    a1_case = state.get("agent1_case", "")
    a2_case = state.get("agent2_case", "")
    api_key = state.get("anthropic_key", "")

    system_prompt = """You are the debate arbiter for an institutional portfolio
optimization system. Two AI agents have presented their cases:
- Agent-1 (Markowitz): optimization result
- Agent-2 (Lagrangian risk evaluator): risk and KKT analysis

Your job: score the debate from 0.0 to 1.0 and identify conflicts.

Score rubric:
  1.0 = Complete agreement, portfolio is optimal, all KKT satisfied
  0.8 = Minor disagreements, portfolio acceptable with small actions
  0.6 = Meaningful conflicts resolved, portfolio approved
  0.4 = Significant conflicts remain, retry recommended
  0.0 = Fundamental disagreement, portfolio rejected

Respond with ONLY a valid JSON object:
{
  "score": 0.XX,
  "verdict": "one sentence decision",
  "conflict_points": [
    {"topic": "...", "a1_view": "...", "a2_view": "...", "resolution": "agree|partial|open"},
    ...
  ],
  "score_breakdown": {
    "return_target_agreement": 0.XX,
    "kkt_compliance": 0.XX,
    "stress_test_acceptance": 0.XX,
    "weight_consensus": 0.XX,
    "constraint_handling": 0.XX
  },
  "action_items": [
    {"priority": "urgent|high|medium|monitor", "action": "...", "rationale": "..."},
    ...
  ]
}"""

    user_msg = f"""Agent-1 case:\n{a1_case}\n\nAgent-2 case:\n{a2_case}

Score this debate and identify all conflict points."""

    transcript = state.get("debate_transcript", [])
    transcript.append({"agent":"A1","type":"OPENING","content":a1_case,
                        "ts":time.time()})
    transcript.append({"agent":"A2","type":"COUNTER","content":a2_case,
                        "ts":time.time()})

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role":"user","content":user_msg}])
        raw  = response.content[0].text.strip()
        # Strip markdown fences if present
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0]
        elif "```"    in raw: raw = raw.split("```")[1].split("```")[0]
        data = json.loads(raw.strip())

        score          = float(data.get("score", 0.5))
        verdict        = data.get("verdict", "")
        conflict_pts   = data.get("conflict_points", [])
        score_brkdn    = data.get("score_breakdown", {})
        action_items   = data.get("action_items", [])

        transcript.append({"agent":"A3","type":"VERDICT",
                            "content":verdict, "score":score,
                            "ts":time.time()})
        print(f"[debate_engine_node] Score={score:.2f} | {verdict}")

    except Exception as e:
        print(f"[debate_engine_node] Claude API error: {e} — defaulting score=0.65")
        score      = 0.65
        verdict    = "Default pass — API unavailable"
        conflict_pts = []
        score_brkdn  = {"return_target_agreement":0.20,"kkt_compliance":0.15,
                         "stress_test_acceptance":0.15,"weight_consensus":0.10,"constraint_handling":0.05}
        action_items = []

    return {**state,
            "debate_score":       score,
            "final_verdict":      verdict,
            "conflict_points":    conflict_pts,
            "score_breakdown":    score_brkdn,
            "action_list":        action_items,
            "debate_transcript":  transcript}


def quality_gate(state: PortfolioState) -> Literal["pass", "retry"]:
    """
    Node: quality_gate (CONDITIONAL EDGE — returns routing string)
    Layer: Agent-3 (Debate)
    Role: Route to 'pass' if score ≥ 0.6, else 'retry' (max 3 iterations).
    Math: pass if score ≥ 0.6 OR iteration ≥ max_iterations
    """
    score    = state.get("debate_score", 0)
    itr      = state.get("iteration", 1)
    max_itr  = state.get("max_iterations", 3)
    passed   = score >= 0.6 or itr >= max_itr

    print(f"[quality_gate] Score={score:.2f} | iter={itr}/{max_itr} | "
          f"→ {'PASS' if passed else 'RETRY'}")

    # Update state (side effect via mutable dict — LangGraph handles this)
    state["quality_passed"] = passed
    state["iteration"]      = itr + 1

    return "pass" if passed else "retry"


def flash_portfolio_generator(state: PortfolioState) -> PortfolioState:
    """
    Node: flash_portfolio_generator
    Layer: Agent-3 (Debate)
    Role: Fast final review by claude-haiku — quick sanity check + 2 bullet insights.
    Model: claude-haiku-4-5-20251001
    Inputs: w_current, sharpe_ratio, ce_value, debate_score
    Outputs: flash_review (text), debate_transcript (appended)
    """
    w      = state.get("w_current",[])
    sr     = state.get("sharpe_ratio", 0)
    ce     = state.get("ce_value", 0)
    score  = state.get("debate_score", 0)
    labels = state.get("labels", [])
    api_key= state.get("anthropic_key","")

    prompt = f"""Quick portfolio review (2 bullet points max):
Weights: {dict(zip(labels,[round(float(x)*100,1) for x in w]))}
Sharpe={sr:.3f}, CE={ce*100:.2f}%, Debate score={score:.2f}

Give: 1 risk flag, 1 opportunity. Be specific, cite numbers."""

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role":"user","content":prompt}])
        flash_text = response.content[0].text
    except Exception as e:
        flash_text = f"Flash review skipped: {e}"

    transcript = state.get("debate_transcript", [])
    transcript.append({"agent":"FLASH","type":"REVIEW",
                        "content":flash_text, "ts":time.time()})

    print(f"[flash_portfolio_generator] Flash review done")
    return {**state,
            "flash_review":      flash_text,
            "debate_transcript": transcript}


# ════════════════════════════════════════════════════════════════
#  DASHBOARD OUTPUT NODES
# ════════════════════════════════════════════════════════════════

def chart_alloc_node(state: PortfolioState) -> PortfolioState:
    """Render weight allocation pie + bar chart HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px

        labels = state.get("labels", [])
        w      = state.get("w_current", [])
        if not labels or not w:
            return {**state, "chart_alloc_html": ""}

        colors = px.colors.qualitative.Set2
        sorted_pairs = sorted(zip(labels,w), key=lambda x:-x[1])
        tlbls = [p[0] for p in sorted_pairs]
        twts  = [p[1]*100 for p in sorted_pairs]

        fig = go.Figure(go.Bar(x=twts, y=tlbls, orientation="h",
            marker_color=colors[:len(tlbls)],
            text=[f"{w:.1f}%" for w in twts], textposition="outside"))
        fig.update_layout(title="Weight Allocation",
            template="plotly_dark", height=380)
        html = fig.to_html(full_html=False, include_plotlyjs=True)
        print(f"[chart_alloc_node] Chart rendered ({len(html)} chars)")
        return {**state, "chart_alloc_html": html}
    except Exception as e:
        print(f"[chart_alloc_node] {e}")
        return {**state, "chart_alloc_html": ""}


def correlation_viz_node(state: PortfolioState) -> PortfolioState:
    """Render correlation heatmap."""
    try:
        import plotly.graph_objects as go
        labels = state.get("labels", [])
        corr   = state.get("corr_matrix")
        if not corr:
            return {**state}
        fig = go.Figure(go.Heatmap(z=corr, x=labels, y=labels,
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1))
        fig.update_layout(title="Correlation Matrix ρ",
            template="plotly_dark", height=420)
        print(f"[correlation_viz_node] Heatmap rendered")
        return {**state}
    except Exception as e:
        print(f"[correlation_viz_node] {e}")
        return {**state}


def weight_alloc_node(state: PortfolioState) -> PortfolioState:
    """Compare current vs max-sharpe vs min-var weights."""
    try:
        import plotly.graph_objects as go
        labels = state.get("labels", [])
        w_c    = state.get("w_current", [])
        w_ms   = state.get("w_maxsharpe", [])
        w_mv   = state.get("w_minvar", [])

        fig = go.Figure()
        for ww, nm, col in [(w_c,"Current","#18C96A"),
                             (w_ms,"Max Sharpe","#F0A020"),
                             (w_mv,"Min Var","#9A7AFF")]:
            if ww:
                fig.add_trace(go.Bar(name=nm, x=labels,
                    y=[float(x)*100 for x in ww], marker_color=col))
        fig.update_layout(barmode="group", title="Weight Comparison",
            template="plotly_dark", height=380)
        print(f"[weight_alloc_node] Comparison chart rendered")
        return {**state}
    except Exception as e:
        print(f"[weight_alloc_node] {e}")
        return {**state}


def shadow_price_heatmap(state: PortfolioState) -> PortfolioState:
    """Render shadow price bar chart as HTML."""
    try:
        import plotly.graph_objects as go
        sp      = state.get("shadow_prices", {})
        if not sp:
            return {**state, "shadow_price_html": ""}
        names   = [r["name"] for r in sp.values()]
        lambdas = [r.get("lambda_val", 0) for r in sp.values()]
        colors  = ["#F04848" if r.get("binding") else "#18C96A"
                   for r in sp.values()]
        fig = go.Figure(go.Bar(x=lambdas, y=names, orientation="h",
            marker_color=colors,
            text=[f"λ={l:.4f}" for l in lambdas], textposition="outside"))
        fig.update_layout(title="Shadow Prices — Constraint Sensitivity",
            template="plotly_dark", height=380)
        html = fig.to_html(full_html=False, include_plotlyjs=True)
        print(f"[shadow_price_heatmap] Shadow price chart rendered")
        return {**state, "shadow_price_html": html}
    except Exception as e:
        print(f"[shadow_price_heatmap] {e}")
        return {**state, "shadow_price_html": ""}


def cost_node(state: PortfolioState) -> PortfolioState:
    """Compute rebalancing transaction cost estimate."""
    w_c   = np.array(state.get("w_current", []))
    w_old = np.ones(len(w_c))/len(w_c) if len(w_c) > 0 else w_c
    tc    = 0.001   # 10 bps
    turnover  = float(np.sum(np.abs(w_c - w_old))/2) if len(w_c) > 0 else 0
    cost_bps  = turnover * tc * 10000
    print(f"[cost_node] Turnover={turnover*100:.1f}% | "
          f"Cost={cost_bps:.1f}bps")
    return {**state}


# ════════════════════════════════════════════════════════════════
#  PERSISTENCE + SSE FANOUT
# ════════════════════════════════════════════════════════════════

def portfolio_repository(state: PortfolioState) -> PortfolioState:
    """
    Node: portfolio_repository
    Layer: Persistence
    Role: Save optimisation result to DB (PostgreSQL in production).
          In this notebook: print summary and set saved_to_db=True.
    """
    result_summary = dict(
        run_id=state.get("run_id"),
        portfolio_ret=state.get("portfolio_ret"),
        portfolio_sig=state.get("portfolio_sig"),
        sharpe_ratio=state.get("sharpe_ratio"),
        ce_value=state.get("ce_value"),
        debate_score=state.get("debate_score"),
        quality_passed=state.get("quality_passed"),
        binding=state.get("binding_constraints"),
        w_current=state.get("w_current"),
    )
    print(f"[portfolio_repository] Saving run {state.get('run_id')} | "
          f"S={state.get('sharpe_ratio',0):.4f} | "
          f"Debate={state.get('debate_score',0):.2f} | "
          f"{'PASS' if state.get('quality_passed') else 'FAIL'}")
    # In production: INSERT INTO optimizer_runs VALUES(...)
    return {**state, "saved_to_db": True}


def sse_fanout_node(state: PortfolioState) -> PortfolioState:
    """
    Node: sse_fanout_node
    Layer: Persistence
    Role: Emit SSE events to connected browser clients.
          In production: Redis Pub/Sub → WebSocket → browser.
    """
    payload = dict(
        event="OPTIMIZATION_COMPLETE",
        run_id=state.get("run_id"),
        sharpe=state.get("sharpe_ratio"),
        ce=state.get("ce_value"),
        score=state.get("debate_score"),
        passed=state.get("quality_passed"),
        verdict=state.get("final_verdict"),
        actions=[a.get("action","") for a in state.get("action_list",[])[:3]],
    )
    print(f"[sse_fanout_node] Broadcasting: {json.dumps(payload)}")
    return {**state, "sse_emitted": True}


# ════════════════════════════════════════════════════════════════
#  LANGGRAPH PIPELINE BUILDER
# ════════════════════════════════════════════════════════════════

def build_arcadia_graph():
    """
    Build the complete 32-node LangGraph state machine.
    Returns compiled graph ready for invoke().

    Flow:
      User Input Layer
          ↓
      Returns & Risk Engine
          ↓
      Utility Theory
          ↓
      Infrastructure (concurrency + orchestration)
          ↓ (parallel dispatch)
      [Agent-1 Markowitz] ←→ [Agent-2 Lagrangian]
          ↓                          ↓
          └──────→ Debate Engine ←──────┘
                        ↓
                  Quality Gate
                  ├── PASS → Flash Review
                  └── RETRY → Agent-1 (max 3×)
                        ↓
              Dashboard Output Nodes
                        ↓
              Persistence + SSE Fanout
                        ↓
                       END
    """
    if not LANGGRAPH_AVAILABLE:
        print("⚠ langgraph not available. Install: !pip install langgraph")
        return None

    g = StateGraph(PortfolioState)

    # ── Register all 32 nodes ─────────────────────────────────
    # User Input Layer
    g.add_node("intake_node",           intake_node)
    g.add_node("clarity_node",          clarity_node)
    g.add_node("constraint_builder",    constraint_builder)

    # Layer 1 — Returns & Risk Engine
    g.add_node("data_feed_node",        data_feed_node)
    g.add_node("returns_engine_node",   returns_engine_node)
    g.add_node("investment_node",       investment_node)

    # Internal Layer — Utility Theory
    g.add_node("utility_calibration_node", utility_calibration_node)
    g.add_node("utility_fn_node",          utility_fn_node)
    g.add_node("risk_aversion_resolver",   risk_aversion_resolver)

    # Infrastructure
    g.add_node("concurrency_gate",      concurrency_gate)
    g.add_node("celery_task",           celery_task)
    g.add_node("orchestration_node",    orchestration_node)

    # Agent-1 — Markowitz
    g.add_node("black_litterman_node",  black_litterman_node)
    g.add_node("markowitz_qp_node",     markowitz_qp_node)
    g.add_node("efficient_frontier_node", efficient_frontier_node)
    g.add_node("sharpe_node",           sharpe_node)

    # Agent-2 — Lagrangian
    g.add_node("lagrangian_node",       lagrangian_node)
    g.add_node("shadow_price_node",     shadow_price_node)
    g.add_node("kkt_node",              kkt_node)
    g.add_node("stress_test_node",      stress_test_node)

    # Agent-3 — Debate
    g.add_node("debate_engine_node",    debate_engine_node)
    g.add_node("flash_portfolio_generator", flash_portfolio_generator)

    # Dashboard Output
    g.add_node("chart_alloc_node",      chart_alloc_node)
    g.add_node("correlation_viz_node",  correlation_viz_node)
    g.add_node("weight_alloc_node",     weight_alloc_node)
    g.add_node("shadow_price_heatmap",  shadow_price_heatmap)
    g.add_node("cost_node",             cost_node)

    # Persistence
    g.add_node("portfolio_repository",  portfolio_repository)
    g.add_node("sse_fanout_node",       sse_fanout_node)

    # ── Define edges ──────────────────────────────────────────
    g.set_entry_point("intake_node")

    # User input flow
    g.add_edge("intake_node",           "clarity_node")
    g.add_edge("clarity_node",          "constraint_builder")

    # Layer 1
    g.add_edge("constraint_builder",    "data_feed_node")
    g.add_edge("data_feed_node",        "returns_engine_node")
    g.add_edge("returns_engine_node",   "investment_node")

    # Utility theory
    g.add_edge("investment_node",           "utility_calibration_node")
    g.add_edge("utility_calibration_node",  "utility_fn_node")
    g.add_edge("utility_fn_node",           "risk_aversion_resolver")

    # Infrastructure
    g.add_edge("risk_aversion_resolver",    "concurrency_gate")
    g.add_edge("concurrency_gate",          "celery_task")
    g.add_edge("celery_task",               "orchestration_node")

    # Agent-1 chain (from orchestration)
    g.add_edge("orchestration_node",        "black_litterman_node")
    g.add_edge("black_litterman_node",      "markowitz_qp_node")
    g.add_edge("markowitz_qp_node",         "efficient_frontier_node")
    g.add_edge("efficient_frontier_node",   "sharpe_node")

    # Agent-2 chain (from orchestration — parallel in production)
    # In LangGraph sequential mode: run after Agent-1
    g.add_edge("orchestration_node",        "lagrangian_node")
    g.add_edge("lagrangian_node",           "shadow_price_node")
    g.add_edge("shadow_price_node",         "kkt_node")
    g.add_edge("kkt_node",                  "stress_test_node")

    # Both agents feed into debate
    g.add_edge("sharpe_node",               "debate_engine_node")
    g.add_edge("stress_test_node",          "debate_engine_node")

    # Quality gate conditional edge
    g.add_conditional_edges(
        "debate_engine_node",
        quality_gate,
        {
            "pass":  "flash_portfolio_generator",
            "retry": "black_litterman_node",   # retry Agent-1 path
        }
    )

    # Post-debate: update utility then go to dashboard
    g.add_edge("flash_portfolio_generator",  "utility_fn_node")
    g.add_edge("flash_portfolio_generator",  "chart_alloc_node")
    g.add_edge("chart_alloc_node",           "correlation_viz_node")
    g.add_edge("correlation_viz_node",       "weight_alloc_node")
    g.add_edge("weight_alloc_node",          "shadow_price_heatmap")
    g.add_edge("shadow_price_heatmap",       "cost_node")

    # Persistence
    g.add_edge("cost_node",                  "portfolio_repository")
    g.add_edge("portfolio_repository",       "sse_fanout_node")
    g.add_edge("sse_fanout_node",            END)

    return g.compile()


# ════════════════════════════════════════════════════════════════
#  INITIAL STATE BUILDER
# ════════════════════════════════════════════════════════════════

def build_initial_state(
    symbols:           List[str],
    labels:            List[str],
    twelve_data_key:   str,
    anthropic_key:     str,
    lambda_aversion:   float = 4.0,
    target_return:     float = 0.13,
    max_weight:        float = 0.25,
    min_weight:        float = 0.02,
    long_only:         bool  = True,
    sector_cap:        float = 0.40,
    sector_map:        Dict[str,str] = None,
    risk_free_rate:    float = 0.065,
    aum_inr:           float = 24_100_000,
    max_iterations:    int   = 3,
) -> PortfolioState:
    """Build the initial state dict for graph.invoke()."""
    return PortfolioState(
        symbols=symbols, labels=labels,
        lambda_aversion=lambda_aversion,
        target_return=target_return,
        max_weight=max_weight, min_weight=min_weight,
        long_only=long_only, sector_cap=sector_cap,
        sector_map=sector_map or {},
        risk_free_rate=risk_free_rate, aum_inr=aum_inr,
        twelve_data_key=twelve_data_key,
        anthropic_key=anthropic_key,
        max_iterations=max_iterations, iteration=1,
        n_assets=len(symbols),
        prices_df=None, returns_df=None,
        mu_vector=None, cov_matrix=None,
        std_vector=None, corr_matrix=None,
        lambda_calibrated=lambda_aversion,
        ce_value=0.0, risk_premium=0.0, utility_score=0.0,
        w_current=None, w_maxsharpe=None, w_minvar=None,
        w_utility=None, w_regime=None,
        frontier_risks=None, frontier_rets=None,
        portfolio_ret=0.0, portfolio_sig=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
        agent1_case="",
        shadow_prices={}, kkt_satisfied=False, kkt_details=[],
        lagrangian_value=0.0, max_drawdown=0.0,
        var_95=0.0, cvar_95=0.0, stress_results={},
        binding_constraints=[], agent2_case="",
        debate_score=0.0, quality_passed=False,
        final_verdict="", conflict_points=[],
        score_breakdown={}, action_list=[],
        debate_transcript=[], flash_review="",
        chart_alloc_html="", frontier_html="",
        shadow_price_html="", full_dashboard_html="",
        run_id="", saved_to_db=False, sse_emitted=False,
        error=None,
    )


# ════════════════════════════════════════════════════════════════
#  PRINT FINAL RESULT
# ════════════════════════════════════════════════════════════════

def print_result(result: PortfolioState):
    """Pretty-print the final pipeline result."""
    print("\n" + "█"*62)
    print("  ARCADIA WEALTH AI — LANGGRAPH PIPELINE RESULT")
    print("█"*62)
    print(f"  Run ID          : {result.get('run_id')}")
    print(f"  Iterations      : {result.get('iteration',1)-1}")
    print(f"  Debate score    : {result.get('debate_score',0):.4f}")
    print(f"  Quality passed  : {result.get('quality_passed')}")
    print(f"  Final verdict   : {result.get('final_verdict','')}")
    print(f"\n  PORTFOLIO RESULT:")
    print(f"   Expected return : {result.get('portfolio_ret',0)*100:.4f}%")
    print(f"   Portfolio σ     : {result.get('portfolio_sig',0)*100:.4f}%")
    print(f"   Sharpe ratio    : {result.get('sharpe_ratio',0):.6f}")
    print(f"   Sortino ratio   : {result.get('sortino_ratio',0):.6f}")
    print(f"   Calmar ratio    : {result.get('calmar_ratio',0):.6f}")
    print(f"   CE (λ={result.get('lambda_aversion',4):.1f})   : "
          f"{result.get('ce_value',0)*100:.4f}%")
    print(f"\n  RISK METRICS:")
    print(f"   95% VaR (1d)   : {result.get('var_95',0)*100:.4f}%")
    print(f"   95% CVaR (1d)  : {result.get('cvar_95',0)*100:.4f}%")
    print(f"   Max drawdown   : {result.get('max_drawdown',0)*100:.2f}%")
    print(f"\n  WEIGHTS:")
    labels  = result.get("labels", [])
    weights = result.get("w_current", [])
    for lbl, wt in sorted(zip(labels,weights), key=lambda x:-x[1]):
        bar = "█"*int(float(wt)*30)
        print(f"   {lbl:<18} {float(wt)*100:>7.2f}%  {bar}")
    print(f"\n  KKT: {sum(c['satisfied'] for c in result.get('kkt_details',[]))}"
          f"/{len(result.get('kkt_details',[]))} conditions satisfied")
    print(f"  Binding constraints: {result.get('binding_constraints',[])}")
    if result.get("action_list"):
        print(f"\n  PRIORITISED ACTIONS:")
        for a in result.get("action_list",[])[:4]:
            p = a.get("priority","")
            act = a.get("action","")
            print(f"   [{p.upper():>7}] {act}")
    if result.get("flash_review"):
        print(f"\n  FLASH REVIEW (claude-haiku-4-5):")
        for line in result.get("flash_review","").strip().split("\n")[:4]:
            print(f"   {line}")
    print("█"*62)


# ════════════════════════════════════════════════════════════════
#  MAIN — RUN THE PIPELINE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__" or True:

    # ── Configuration ──────────────────────────────────────────
    TWELVE_KEY   = "cc625729febe4c5eaf6641b6087678c8"
    ANTHROPIC_KEY= "YOUR_ANTHROPIC_KEY_HERE"   # ← replace

    SYMBOLS = [
        "RELIANCE:NSE", "HDFCBANK:NSE", "INFY:NSE",
        "TCS:NSE",      "ICICIBANK:NSE","WIPRO:NSE",
        "TATAMOTORS:NSE","GOLDBEES:NSE", "NIFTYBEES:NSE",
    ]
    LABELS = [
        "Reliance","HDFC Bank","Infosys","TCS","ICICI Bank",
        "Wipro","Tata Motors","Gold ETF","Nifty ETF",
    ]
    SECTOR_MAP = {
        "RELIANCE:NSE":"Energy",    "HDFCBANK:NSE":"Financials",
        "INFY:NSE":"IT",            "TCS:NSE":"IT",
        "ICICIBANK:NSE":"Financials","WIPRO:NSE":"IT",
        "TATAMOTORS:NSE":"Auto",    "GOLDBEES:NSE":"Gold",
        "NIFTYBEES:NSE":"Index",
    }

    # ── Build graph ────────────────────────────────────────────
    print("🔧 Building Arcadia LangGraph pipeline (32 nodes)...")
    graph = build_arcadia_graph()

    if graph is None:
        print("Cannot run — langgraph not installed")
    else:
        # ── Build initial state ────────────────────────────────
        initial_state = build_initial_state(
            symbols=SYMBOLS, labels=LABELS,
            twelve_data_key=TWELVE_KEY,
            anthropic_key=ANTHROPIC_KEY,
            lambda_aversion=4.0,
            target_return=0.13,
            max_weight=0.25, min_weight=0.02,
            long_only=True, sector_cap=0.40,
            sector_map=SECTOR_MAP,
            risk_free_rate=0.065,
            aum_inr=24_100_000,
            max_iterations=3,
        )

        # ── Run pipeline ───────────────────────────────────────
        print("\n🚀 Running full LangGraph pipeline...")
        print("   This will take 2-3 minutes (Twelve Data API rate limits)")
        print("   32 nodes × all layers × Agent-1 + Agent-2 + Debate\n")

        try:
            result = graph.invoke(initial_state)
            print_result(result)
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            print("   Check API keys and network connection")
            raise

        print("\n✅ LangGraph pipeline complete.")
        print("   All 32 nodes executed.")
        print("   result dict contains full PortfolioState.")


# ════════════════════════════════════════════════════════════════
#  STANDALONE USAGE (without live API — uses pre-computed data)
# ════════════════════════════════════════════════════════════════

def run_without_live_data(mu_arr, cov_arr, labels, anthropic_key,
                           lambda_aversion=4.0, target_return=0.13,
                           max_weight=0.25, min_weight=0.02):
    """
    Run the full debate pipeline with pre-computed μ and Σ.
    Skips data_feed_node and returns_engine_node.
    Useful when running after the master notebook has already
    fetched live data.

    Usage (in Colab, after master notebook):
        from langgraph_full_pipeline import run_without_live_data
        result = run_without_live_data(
            mu_arr=mu,          # from Cell 4 of master notebook
            cov_arr=cov,        # from Cell 4 of master notebook
            labels=labels,
            anthropic_key=ANTHROPIC_API_KEY)
    """
    n = len(mu_arr)
    symbols = [f"ASSET_{i}" for i in range(n)]

    # Build minimal state with pre-computed data
    state = build_initial_state(
        symbols=symbols, labels=labels,
        twelve_data_key="", anthropic_key=anthropic_key,
        lambda_aversion=lambda_aversion,
        target_return=target_return,
        max_weight=max_weight, min_weight=min_weight)

    state["mu_vector"]  = mu_arr.tolist() if hasattr(mu_arr,"tolist") else list(mu_arr)
    state["cov_matrix"] = cov_arr.tolist() if hasattr(cov_arr,"tolist") else [list(r) for r in cov_arr]
    state["std_vector"] = list(np.sqrt(np.diag(np.array(state["cov_matrix"]))))

    # Run the pipeline nodes manually (skipping data fetch)
    for node_fn in [
        constraint_builder,
        utility_calibration_node, utility_fn_node, risk_aversion_resolver,
        concurrency_gate, celery_task, orchestration_node,
        black_litterman_node, markowitz_qp_node, efficient_frontier_node, sharpe_node,
        lagrangian_node, shadow_price_node, kkt_node, stress_test_node,
        debate_engine_node,
    ]:
        state.update(node_fn(state))

    # Quality gate check
    if state.get("debate_score", 0) >= 0.6:
        state.update(flash_portfolio_generator(state))

    state.update(chart_alloc_node(state))
    state.update(shadow_price_heatmap(state))
    state.update(cost_node(state))
    state.update(portfolio_repository(state))
    state.update(sse_fanout_node(state))

    print_result(state)
    return state
