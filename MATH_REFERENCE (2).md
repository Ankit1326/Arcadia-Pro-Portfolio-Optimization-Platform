# Arcadia Wealth AI — Complete Math Reference
### Every formula used in the project, with derivation notes and implementation hints

---

## 1. Returns & Risk Engine

### 1.1 Log returns
```
rᵢₜ = ln(Pᵢₜ / Pᵢ,ₜ₋₁)

Reason for log over simple: log returns are additive across time, normally distributed,
and bounded below at -∞ (preventing negative prices). Used for all covariance estimation.
```

### 1.2 Expected return — historical mean
```
μᵢ = (1/T) Σₜ rᵢₜ  ×  252        (annualised)

T = number of daily observations. Multiply by 252 (NSE trading days/year).
Implementation: mu = returns.mean() * 252
```

### 1.3 Expected return — CAPM
```
μᵢ = Rf + βᵢ × (E[Rm] − Rf)

βᵢ = Cov(rᵢ, Rm) / Var(Rm) = Σᵢₘ / σ²ₘ

Rf  = risk-free rate (India 10Y G-Sec, currently 6.5%)
Rm  = Nifty 50 index return
E[Rm] − Rf = equity risk premium (historically ~5-7% for India)
```

### 1.4 Expected return — Black-Litterman
```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ × [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

Π   = λ_mkt × Σ × w_mkt    (implied equilibrium returns)
P   = pick matrix (k×n, where k = number of views)
Q   = view return vector (k×1)
Ω   = view uncertainty matrix (k×k, diagonal)
τ   = uncertainty scalar (typically 1/T ≈ 0.05)

Blend: μ_final = α × μ_BL + (1−α) × μ_historical   (α=0.70 default)
```

### 1.5 Covariance matrix (annualised)
```
Σᵢⱼ = Cov(rᵢ, rⱼ)_daily × 252

Implementation: cov = returns.cov() * 252

Positive semi-definite check: all eigenvalues ≥ 0
Shrinkage (Ledoit-Wolf) for small samples:
    Σ_shrunk = (1-δ)×Σ_sample + δ×μ_target×I
    δ optimal via Ledoit-Wolf lemma
```

### 1.6 Correlation matrix
```
ρᵢⱼ = Σᵢⱼ / (σᵢ × σⱼ)

Implementation: corr = returns.corr()  # Pearson, automatically normalized
```

---

## 2. Markowitz Mean-Variance Optimization

### 2.1 Core optimization problem
```
min   wᵀΣw                    (minimize portfolio variance)
w

subject to:
    wᵀμ ≥ r*                  (return floor constraint)
    Σᵢwᵢ = 1                  (budget constraint, fully invested)
    wᵢ ≥ 0                    (long-only, or wᵢ ≥ −0.20 for short)
    wᵢ ≤ w_max                (single-asset cap, e.g. 0.25)
    wᵢ ≥ w_min                (minimum position, e.g. 0.02)
    Σᵢ∈Sₖ wᵢ ≤ c_k           (sector cap for each sector k)

This is a Quadratic Program (QP). Solved via CVXPY → OSQP solver.
Solver convergence tolerance: ftol = 1e-12, maxiter = 1000
```

### 2.2 Portfolio statistics
```
Expected return:   E[rp] = wᵀμ
Portfolio variance: σ²p  = wᵀΣw
Portfolio std dev:  σp   = √(wᵀΣw)
Sharpe ratio:       S    = (E[rp] − Rf) / σp
Information ratio:  IR   = (E[rp] − E[rb]) / TE
Tracking error:     TE   = √((w−wb)ᵀΣ(w−wb))
```

### 2.3 Efficient frontier construction
```
Algorithm:
    1. Set r_min = min(μ) × 1.01, r_max = max(μ) × 0.98
    2. Create targets = linspace(r_min, r_max, 80)
    3. For each target r*:
           solve: min wᵀΣw  s.t.  wᵀμ = r*, Σwᵢ=1, w≥0, w≤w_max
           record: (σp(w*), E[r]) if solver converged
    4. Plot curve: x=σp, y=E[r]

The minimum variance point is the leftmost feasible point on the frontier.
The maximum Sharpe point is where the Capital Market Line (CML) is tangent to the frontier.
```

### 2.4 Capital Market Line
```
CML: E[r] = Rf + [(E[rT] − Rf) / σT] × σ

Where (E[rT], σT) is the tangency (max Sharpe) portfolio.
The slope of the CML = Sharpe ratio of the tangency portfolio.
All points on the CML above the tangency are achieved by leveraging the tangency portfolio.
```

### 2.5 Maximum Sharpe portfolio
```
max   (wᵀμ − Rf) / √(wᵀΣw)
 w

This is a non-linear fractional program. Convert to QP:
    Let y = w/κ where κ = (wᵀμ − Rf)
    Then maximize yᵀ(μ−Rf) s.t. yᵀΣy = 1, yᵢ ≥ 0

Equivalently solved numerically:
    min   −(wᵀμ − Rf) / √(wᵀΣw)     via scipy.optimize.minimize SLSQP
```

---

## 3. Utility Theory

### 3.1 Mean-variance utility (CARA approximation)
```
U(w) = E[rp] − (λ/2) × σ²p
     = wᵀμ − (λ/2) × wᵀΣw

λ = risk aversion coefficient
λ = 0.5 → near risk-neutral (aggressive speculator)
λ = 4.0 → balanced
λ = 8.0 → conservative
λ = 12  → ultra-conservative (capital preservation)
```

### 3.2 Certainty Equivalent
```
CE = E[rp] − (λ/2) × σ²p

Interpretation: The guaranteed (risk-free) return that the investor
is indifferent to vs the risky portfolio.

Risk premium: RP = E[rp] − CE = (λ/2) × σ²p

At λ=4, σ=8.2%:  RP = (4/2) × 0.082² × 10000 = 1.35% annually
```

### 3.3 Indifference curve
```
For fixed utility level U₀, the indifference curve in (σ, E[r]) space is:
    r = U₀ + (λ/2) × σ²

This is a parabola opening upward. Higher U₀ = higher indifference curve.
The portfolio lies on the efficient frontier where its utility is maximized.
```

### 3.4 Utility-optimal portfolio
```
max  wᵀμ − (λ/2) × wᵀΣw
 w

s.t. same constraints as Markowitz

As λ→0: approaches maximum return portfolio (full concentration)
As λ→∞: approaches minimum variance portfolio (maximum diversification)
```

### 3.5 CRRA utility (power utility)
```
U(W) = W^(1−γ) / (1−γ)     if γ ≠ 1
U(W) = ln(W)                if γ = 1 (log utility)

γ = coefficient of relative risk aversion
γ ≈ 2λσ² at small wealth changes (connects to MV utility)
```

### 3.6 CARA utility (exponential)
```
U(W) = −exp(−α×W) / α

α = absolute risk aversion coefficient
Constant absolute risk aversion regardless of wealth level.
More applicable for institutional investors managing fixed AUM.
```

---

## 4. Lagrangian Optimization & Shadow Prices

### 4.1 Full Lagrangian
```
L(w, λ₁, λ₂, μ, ν) = wᵀΣw
                      − λ₁(wᵀμ − r*)           [return floor]
                      − λ₂(Σwᵢ − 1)            [budget constraint]
                      − Σᵢ μᵢ(wᵢ − w_max)       [upper bound]
                      − Σᵢ νᵢ(w_min − wᵢ)       [lower bound]

λ₁, λ₂ = Lagrange multipliers (unrestricted sign for equality constraints)
μᵢ, νᵢ = KKT multipliers (≥ 0 for inequality constraints)
```

### 4.2 KKT conditions (necessary & sufficient for convex QP)

**Condition 1: Stationarity (∂L/∂w = 0)**
```
2Σw* − λ₁μ − λ₂1 − μ + ν = 0

||∇_w L|| ≤ tolerance (e.g. 1e-6) at optimal w*
```

**Condition 2: Primal feasibility**
```
wᵀμ ≥ r*                   (return floor satisfied)
Σwᵢ = 1                     (fully invested)
wᵢ ≥ w_min for all i        (minimum weights)
wᵢ ≤ w_max for all i        (maximum weights)
```

**Condition 3: Dual feasibility**
```
λ₁ ≥ 0  (return floor is inequality constraint)
μᵢ ≥ 0 for all i
νᵢ ≥ 0 for all i
```

**Condition 4: Complementary slackness**
```
λ₁ × (wᵀμ − r*) = 0
μᵢ × (wᵢ − w_max) = 0  for all i
νᵢ × (w_min − wᵢ) = 0  for all i

Interpretation:
- If λ₁ > 0: return constraint is BINDING (active)
- If λ₁ = 0: return constraint is SLACK (not limiting the solution)
- If μᵢ > 0: asset i is at its upper weight cap (binding)
- If νᵢ > 0: asset i is at its lower weight floor (binding)
```

### 4.3 Shadow price interpretation
```
Shadow price λᵢ = ∂(optimal objective) / ∂(constraint rhs)

Example:
λ₁ = 0.31 means: if return floor is relaxed by 1%, portfolio variance
                  decreases by 0.31 units.

Financial interpretation: λᵢ is the marginal cost (in variance units)
of tightening constraint i by one unit.

Binding constraint (λ > 0): The constraint is limiting the solution.
                              Relaxing it improves the objective.
Slack constraint  (λ = 0): The constraint is not active. Relaxing it
                              has no effect on the optimal solution.
```

### 4.4 Finite-difference shadow price estimation
```
λᵢ ≈ [f*(rhs + δ) − f*(rhs)] / δ     (forward difference)
λᵢ ≈ [f*(rhs + δ) − f*(rhs − δ)] / 2δ  (central difference, more accurate)

δ = 0.001 (0.1% perturbation)

f*(rhs) = optimal variance at given constraint rhs

Implementation:
    base_var = w_opt @ Sigma @ w_opt
    res_perturbed = solve_qp(target_return - delta)
    shadow_price = (base_var - res_perturbed.x @ Sigma @ res_perturbed.x) / delta
```

---

## 5. VaR & CVaR

### 5.1 Parametric VaR (Gaussian assumption)
```
VaR_α = −(μ_daily + z_α × σ_daily × √T)

where:
    z_α = standard normal quantile:
          α=90%  → z = 1.282
          α=95%  → z = 1.645
          α=99%  → z = 2.326
          α=99.9%→ z = 3.090
    T   = horizon in trading days (1, 10, 21)
    μ_daily = portfolio daily mean return
    σ_daily = portfolio daily standard deviation

VaR is expressed as a positive number representing maximum loss.
```

### 5.2 Historical simulation VaR
```
VaR_α = −Quantile(historical_returns, 1−α)

Sort all T historical daily portfolio returns in ascending order.
VaR at 95% = the (T × 5%)th worst return.

Advantages: Captures skewness, kurtosis, and fat tails automatically.
Disadvantages: Requires sufficient history; backwards-looking.
```

### 5.3 Monte Carlo VaR
```
1. Generate N=10,000 random return paths:
       r_sim ~ N(μ, Σ)   (or Student-t for fat tails)
2. Compute portfolio return for each path:
       rp_i = wᵀ r_sim_i
3. VaR_α = −Quantile({rp_i}, 1−α)

Advantages: Most flexible; can incorporate non-linear positions.
```

### 5.4 VaR scaling (√T rule)
```
VaR_T = VaR_1day × √T

Assumes: i.i.d. daily returns (serial independence)
Valid for: Parametric VaR, short horizons (T ≤ 21)
Breaks down for: T > 1 month (autocorrelation matters)
```

### 5.5 CVaR / Expected Shortfall
```
CVaR_α = −E[rp | rp ≤ −VaR_α]

Parametric (Gaussian):
    CVaR_α = VaR_α + σ_daily × φ(z_α) / (1−α)
    where φ = standard normal PDF

ES multipliers:
    α=90%:   CVaR/VaR ≈ 1.755
    α=95%:   CVaR/VaR ≈ 2.063
    α=99%:   CVaR/VaR ≈ 2.665
    α=99.9%: CVaR/VaR ≈ 3.368

CVaR is always ≥ VaR. CVaR is coherent (subadditive); VaR is not.
CVaR is the Basel III / FRTB standard for trading book capital.
```

### 5.6 Marginal VaR (Euler decomposition)
```
MVaR_i = wᵢ × ∂VaR_p / ∂wᵢ
        ≈ wᵢ × (Σw)_i / σp × z_α / √252

Euler's theorem: Σᵢ MVaR_i = VaR_p   (exact decomposition)

Interpretation: MVaR_i is how much VaR would change if wᵢ increased by 1%.
The asset with the highest MVaR is the dominant tail-risk driver.

% contribution: MVaR_i / VaR_p × 100
```

### 5.7 Drawdown metrics
```
Drawdown_t = (NAV_t − Peak_t) / Peak_t
Max drawdown = min_t(Drawdown_t)

Calmar ratio = Annualised return / |Max drawdown|
    Target for institutional: Calmar ≥ 1.0

Sortino ratio = (E[rp] − Rf) / σ_downside
    σ_downside = √(E[min(r − MAR, 0)²])
    MAR = Minimum Acceptable Return (often = Rf)
    Sortino > Sharpe means upside volatility is high (good)
```

---

## 6. Macro Regime Detection (HMM)

### 6.1 Hidden Markov Model
```
States: S = {Bull, Late-cycle, Bear, Recovery}
Observations: O_t = vector of macro indicators at time t

Model parameters (θ):
    π    = initial state probability vector (4×1)
    A    = transition matrix (4×4)
         A[i,j] = P(S_{t+1}=j | S_t=i)
    B    = emission probability (Gaussian)
         P(O_t | S_t=k) = N(μ_k, Σ_k)

Training: Baum-Welch algorithm (EM)
    E-step: forward-backward algorithm to compute P(S_t | O_1:T)
    M-step: update μ_k, Σ_k, A, π to maximize likelihood

Inference: Viterbi algorithm for most likely state sequence
    δ_t(k) = max_{s₁..sₜ₋₁} P(s₁,..,s_{t-1}, S_t=k, O_1:t | θ)
```

### 6.2 Feature engineering for HMM
```
Input features (standardized to zero mean, unit variance):
    x₁ = India Manufacturing PMI (monthly)
    x₂ = India VIX (daily, smoothed 21d MA)
    x₃ = 2Y-10Y yield curve spread (daily)
    x₄ = FII net equity flows (monthly, INR Cr)
    x₅ = Nifty 50 P/E ratio (daily)
    x₆ = India CPI YoY (monthly)
    x₇ = GDP growth estimate (quarterly, interpolated)
    x₈ = RBI repo rate (daily, step function)
    x₉ = Credit growth YoY (monthly)
    x₁₀= Nifty 50 30d realized volatility (daily)

Preprocessing:
    - Fill monthly/quarterly series forward to daily frequency
    - Apply 5-day smoothing to reduce noise
    - Standardize: (x − μ) / σ using rolling 252d window
```

### 6.3 Regime-weighted expected returns
```
μ*(t) = Σᵣ P(regime=r | O₁:t) × μ(r)

Where μ(r) = historical mean return of each asset during regime r

This is the probability-weighted expected return vector used in the
regime-adjusted Markowitz optimization:
    min wᵀΣw  s.t.  wᵀμ* ≥ r*

Effect: Assets that perform well in high-probability regimes get higher μ*
        and thus higher optimal weight.
```

### 6.4 Transition matrix calibration
```
A_calibrated[i,j] = (count of regime transitions from i to j) / (count of time in regime i)

Using 10+ years of historical data.
Updated: weekly (incremental EM update)

Forward simulation (Monte Carlo regime paths):
    P(S_{t+k} = j | S_t = i) = (A^k)[i,j]
    
    For 30-day forward probability:
        π_{t+30} = A³⁰ × π_t

    Run 500 Monte Carlo paths by sampling from A at each step.
```

---

## 7. Factor Model

### 7.1 Fama-French 5-factor + extensions
```
rᵢ − Rf = αᵢ + Σⱼ βᵢⱼ Fⱼ + εᵢ

Factors:
    F₁ = Rm − Rf        (market risk premium)
    F₂ = SMB            (small minus big, size factor)
    F₃ = HML            (high minus low, value factor)
    F₄ = RMW            (robust minus weak, profitability)
    F₅ = CMA            (conservative minus aggressive, investment)
    F₆ = MOM            (12-1 month price momentum)
    F₇ = QMJ            (quality minus junk)
    F₈ = BAB            (betting against beta, low-vol factor)
    F₉ = Carry          (dividend yield vs risk-free)
    F₁₀= LIQ            (Amihud liquidity measure)

Portfolio factor exposure: β_p = Σᵢ wᵢ × βᵢ
Factor-explained variance: σ²_explained = βᵀ F_cov β
Idiosyncratic variance: σ²_idio = σ²_total − σ²_explained
```

### 7.2 Beta calculation
```
βᵢ = Cov(rᵢ, Rm) / Var(Rm)

Estimation: OLS regression of excess asset returns on excess market returns
    rᵢ − Rf = αᵢ + βᵢ(Rm − Rf) + εᵢ
    using 252 daily observations (1 year)

For portfolio: β_p = wᵀβ   (vector dot product)
```

---

## 8. Options & Greeks

### 8.1 Black-Scholes pricing
```
Call: C = S×N(d₁) − K×e^(−rT)×N(d₂)
Put:  P = K×e^(−rT)×N(−d₂) − S×N(−d₁)

d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T

S = spot price, K = strike, r = risk-free rate
T = time to expiry in years, σ = implied volatility
N(·) = standard normal CDF
```

### 8.2 Greeks
```
Delta:  Δ = ∂C/∂S = N(d₁)              (call)
                   = N(d₁) − 1          (put)
        Range: [0,1] for calls, [-1,0] for puts

Gamma:  Γ = ∂²C/∂S² = φ(d₁) / (S×σ×√T)
        Same for calls and puts. Maximum at ATM.

Theta:  Θ = −∂C/∂T = −S×φ(d₁)×σ/(2√T) − r×K×e^(−rT)×N(d₂)   (call)
        Expressed per day: divide by 365 or 252

Vega:   V = ∂C/∂σ = S×φ(d₁)×√T
        Per 1% IV change: divide by 100

Rho:    ρ = ∂C/∂r = K×T×e^(−rT)×N(d₂)   (call)
                   = −K×T×e^(−rT)×N(−d₂) (put)

φ(·) = standard normal PDF = (1/√2π)×e^(−x²/2)
```

### 8.3 Implied volatility (Newton-Raphson)
```
IV_0 = initial guess (e.g. historical vol or ATM approximation)
IV_{n+1} = IV_n − (BS_price(IV_n) − Market_price) / Vega(IV_n)

Converges in 5-10 iterations for typical inputs.
Tolerance: |BS_price(IV) − Market_price| < 0.001
```

### 8.4 Put-call skew
```
Skew = IV(25Δ put) − IV(25Δ call)

Negative skew: OTM puts more expensive than OTM calls
             → market paying for downside protection
             → bearish demand (common in equity markets)

25Δ put: the strike K where Δ_put = −0.25
25Δ call: the strike K where Δ_call = +0.25
```

### 8.5 Volatility risk premium
```
VRP = IV_30d − RV_30d

IV_30d = 30-day at-the-money implied volatility
RV_30d = realized volatility over past 30 days

VRP > 0 → options overpriced vs realized → sell vol signal
VRP < 0 → options underpriced vs realized → buy vol signal
Historical VRP for NSE: typically +2% to +4%
```

---

## 9. Liquidity & Market Impact

### 9.1 Amihud illiquidity measure
```
ILLIQ_i = (1/T) Σₜ |rᵢₜ| / Vol_iₜ

|rᵢₜ| = absolute daily return of asset i on day t
Vol_iₜ = daily trading volume in value terms (INR)

Higher ILLIQ → more price impact per unit of volume → less liquid
Typical NSE large-cap: ILLIQ < 0.01
Typical NSE mid-cap: ILLIQ 0.01 − 0.05
```

### 9.2 Kyle's lambda (price impact coefficient)
```
ΔP_t = λ × OrderFlow_t + ε_t

λ = Cov(ΔP, OrderFlow) / Var(OrderFlow)

Higher λ → larger price move per unit of net order flow → less liquid
Estimated via OLS regression of daily price changes on signed volume.
```

### 9.3 Square-root market impact model
```
ΔP/P = σ × √(Q/ADV) × sign(Q)

σ   = daily volatility of the asset
Q   = order size in shares
ADV = average daily volume in shares

Impact in bps: ΔP/P × 10000

Example: σ=1.4%, Q/ADV=5% → Impact = 1.4% × √(0.05) = 0.31%  = 31 bps
```

### 9.4 Almgren-Chriss optimal execution
```
Minimize: E[Cost] + λ_risk × Var[Cost]

E[Cost] = Σₜ (η × vₜ²/τ + γ × xₜ × vₜ)     (market impact)
Var[Cost] = σ² × Σₜ xₜ² × τ                   (execution risk)

Where:
    vₜ = shares traded in interval t
    xₜ = remaining shares at time t (inventory)
    τ  = length of trading interval
    η  = temporary impact coefficient
    γ  = permanent impact coefficient
    λ_risk = risk aversion in execution

Optimal trajectory: xₜ = X × sinh(κ(T−t)) / sinh(κT)
    κ = √(λ_risk × σ² / η)
    X = total shares to liquidate

VWAP special case: λ_risk = 0 → uniform execution across intervals
```

---

## 10. Tax Calculations (India)

### 10.1 Capital gains classification
```
Short-Term Capital Gain (STCG):
    Holding period < 12 months for listed equity
    Tax rate: 15% (Section 111A of Income Tax Act)
    Surcharge + cess applies

Long-Term Capital Gain (LTCG):
    Holding period ≥ 12 months for listed equity
    Exemption: First ₹1,00,000 per FY is tax-free
    Tax rate: 10% on gains above ₹1,00,000 (Section 112A)
    No indexation benefit for equity
    
Special: Gold ETF, Debt ETF, REITs have different rules (3 years for LTCG)
```

### 10.2 FIFO lot selection (default)
```
For a sale of Q shares:
    1. Sort lots by purchase_date ascending (oldest first)
    2. Deplete earliest lot first
    3. If lot_qty ≥ remaining_Q: fully deplete this lot, stop
    4. If lot_qty < remaining_Q: fully deplete, reduce remaining_Q, continue

Tax on each lot:
    if purchase_date ≥ 12 months ago: gain × 10% (LTCG)
    else: gain × 15% (STCG)
```

### 10.3 LIFO lot selection (tax-optimal in some cases)
```
Same as FIFO but sort by purchase_date descending (newest first).

When LIFO saves more tax:
    - When newest lots have lower gains (bought at higher prices = smaller gain)
    - When newest lots are LTCG eligible and STCG rate applies to oldest lots
    - When you have a recent lot at a loss (LIFO harvests the loss first)

India law: Investors can specify lot selection. Default is FIFO per SEBI guidelines.
```

### 10.4 Tax-loss harvesting
```
Net tax liability = LTCG_tax + STCG_tax − Loss_offset

Rules:
    STCG losses offset STCG gains first, then LTCG gains
    LTCG losses offset LTCG gains only
    Carry forward: losses can be carried forward 8 years

Harvest opportunity: Sell a loss position before year-end to crystallize the loss.
    Tax saving = loss × applicable_rate
    Net saving after transaction cost = tax_saving − brokerage_STT

Wash sale: Not applicable in India (unlike US). Can immediately rebuy.
```

---

## 11. Performance Attribution (Brinson-Hood-Beebower)

```
Active return = E[rp] − E[rb]   (alpha vs benchmark)

Active return decomposed into:
    Allocation effect  = Σᵢ (wᵢp − wᵢb)(rᵢb − rb)
    Selection effect   = Σᵢ wᵢb(rᵢp − rᵢb)
    Interaction effect = Σᵢ (wᵢp − wᵢb)(rᵢp − rᵢb)

Where:
    wᵢp = portfolio weight of sector i
    wᵢb = benchmark weight of sector i
    rᵢp = portfolio return within sector i
    rᵢb = benchmark return within sector i
    rb  = total benchmark return

Sum check: Allocation + Selection + Interaction = Active return
```

---

## 12. News Sentiment Scoring

```
Sentiment pipeline:
    1. Fetch headline + abstract
    2. Tokenize and clean (remove stopwords, lowercase)
    3. Score via VADER or FinBERT (financial domain model)
    4. Normalize to [-1, +1] scale

VADER: Rule-based. Fast. Good for short texts.
FinBERT: Transformer fine-tuned on financial news. More accurate.

Portfolio sentiment score:
    S_portfolio = Σᵢ wᵢ × S_asset_i × relevance_score_i

relevance_score: TF-IDF similarity between headline and asset description.
Range: [0, 1]. Only include articles with relevance > 0.3.
```

---

## 13. Key References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77-91.
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*, 48(5), 28-43.
- Fama, E. & French, K. (1993). Common risk factors in stock and bond returns. *JFE*, 33, 3-56.
- Fama, E. & French, K. (2015). A five-factor asset pricing model. *JFE*, 116(1), 1-22.
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.).
- Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*, 57(2), 357-384.
- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5-39.
- Karush, W. (1939). Minima of functions of several variables with inequalities as side conditions. MSc thesis. / Kuhn, H.W. & Tucker, A.W. (1951). Nonlinear programming. *Proc. 2nd Berkeley Symp. Math. Stat. Prob.*
- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *JFMA*, 5(1), 31-56.
- Kyle, A.S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
- Brinson, G.P., Hood, L.R. & Beebower, G.L. (1986). Determinants of portfolio performance. *FAJ*, 42(4), 39-44.
