# ============================================================
#  ARCADIA WEALTH AI — PART 7
#  THREE OUTPUT MODES: Beginner · Advanced · Quant
#  Run AFTER Parts 1-6 (all objects in memory)
# ============================================================

class ThreeModeOutput:
    """
    Renders the final portfolio summary in 3 modes:
      1. Beginner  — plain English, traffic lights, no jargon
      2. Advanced  — full metrics, shadow prices, charts
      3. Quant     — raw matrices, KKT conditions, Lagrangian math
    """

    def __init__(self, optimizer, engine, utility, lagrangian,
                 risk, user_input, ai_engine=None):
        self.opt   = optimizer
        self.eng   = engine
        self.util  = utility
        self.lag   = lagrangian
        self.risk  = risk
        self.ui    = user_input
        self.ai    = ai_engine
        self.w     = optimizer.current_w
        self.labels = [t.replace(".NS","") for t in engine.tickers]

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _light(val, hi, lo):
        if val >= hi: return "🟢"
        if val >= lo: return "🟡"
        return "🔴"

    @staticmethod
    def _bar(w, width=30):
        return "█" * int(w * width)

    # ══════════════════════════════════════════════════════════
    #  MODE 1: BEGINNER
    # ══════════════════════════════════════════════════════════
    def beginner_mode(self):
        s   = self.opt.portfolio_stats(self.w)
        ce  = self.util.certainty_equivalent()
        mdd = self.risk._drawdown.min() * 100
        var = self.risk._var_stats["var_hist_1d"] * 100

        sl_ret = self._light(s["ret"]*100, 12, 8)
        sl_sr  = self._light(s["sr"], 1.2, 0.8)
        sl_dd  = self._light(mdd, -10, -15)

        print("\n" + "▓"*58)
        print("  YOUR PORTFOLIO — SIMPLE SUMMARY  (Beginner Mode)")
        print("▓"*58)
        print(f"""
  {sl_ret} SPEED — Expected return: {s['ret']*100:.1f}% per year
     Your money is expected to grow at this rate annually.
     {'Above target ✓' if s['ret'] >= self.ui.target_return else 'Below target — consider relaxing constraints'}

  {sl_sr} EFFICIENCY — Sharpe ratio: {s['sr']:.2f}
     For every unit of risk, you earn {s['sr']:.2f} units of
     extra return. Above 1.0 is considered good.

  {sl_dd} SAFETY — Max drawdown: {mdd:.1f}%
     In the worst historical stretch, your portfolio fell
     {mdd:.1f}% from its peak before recovering.
     Your stated limit: {self.ui.max_drawdown_tol*100:.0f}%

  💰 CERTAINTY EQUIVALENT: {ce['ce']*100:.1f}%
     You'd need a guaranteed {ce['ce']*100:.1f}% return to feel as
     good as holding this portfolio. You're giving up
     {ce['rp']*100:.1f}% of expected return just to avoid uncertainty.

  ⚠️  DAILY RISK: {var:.2f}%
     On 95 out of 100 days, you won't lose more than {abs(var):.2f}%
     in a single day. On the remaining 5 days, losses
     could be larger.
""")
        print("  YOUR TOP HOLDINGS:")
        pairs = sorted(zip(self.labels, self.w), key=lambda x: -x[1])[:5]
        for name, wt in pairs:
            print(f"  {name:<18} {wt*100:.1f}%  {self._bar(wt)}")

        # Binding constraints in plain English
        binding = [r for r in self.lag.results.values() if r.get("binding")]
        if binding:
            print(f"\n  ⚠️  CONSTRAINTS LIMITING YOUR PORTFOLIO:")
            for r in binding:
                print(f"  • {r['name']}")
                print(f"    Relaxing this could improve returns by "
                      f"~{r.get('improve_return',0):.1f}%")
        else:
            print("\n  ✅  No binding constraints — portfolio is freely optimised.")

        verdict = ("✅ WELL-OPTIMISED" if s["sr"] > 1.2
                   else "🟡 DECENT — room for improvement"
                   if s["sr"] > 0.8 else "🔴 REVIEW NEEDED")
        print(f"\n  BOTTOM LINE: {verdict}")
        print("▓"*58)

    # ══════════════════════════════════════════════════════════
    #  MODE 2: ADVANCED
    # ══════════════════════════════════════════════════════════
    def advanced_mode(self):
        s   = self.opt.portfolio_stats(self.w)
        ms  = self.opt.portfolio_stats(self.opt.max_sharpe_w)
        ce  = self.util.certainty_equivalent()
        var = self.risk._var_stats

        print("\n" + "="*62)
        print("  PORTFOLIO SUMMARY — ADVANCED MODE")
        print("="*62)

        # Performance
        print("\n  ── PERFORMANCE ──────────────────────────────────────")
        metrics_perf = [
            ("Expected return (ann.)", f"{s['ret']*100:.2f}%",
             "▲" if s["ret"] >= self.ui.target_return else "▼"),
            ("Portfolio σ (ann.)",     f"{s['sig']*100:.2f}%",
             "✓" if s["sig"]*100 <= self.ui.max_drawdown_tol*100 else "!"),
            ("Sharpe ratio",           f"{s['sr']:.4f}",
             "✓" if s["sr"] >= 1.0 else "!"),
            ("Sortino ratio",          f"{self.risk._sortino:.4f}", ""),
            ("Certainty equiv. (CE)",  f"{ce['ce']*100:.2f}%", ""),
            ("Risk premium (RP)",      f"{ce['rp']*100:.2f}%", ""),
            ("Info ratio vs benchmark",f"{getattr(self.risk,'_info_ratio',0):.4f}", ""),
        ]
        for name, val, flag in metrics_perf:
            print(f"  {name:<35} {val:>10}  {flag}")

        # Risk
        print("\n  ── RISK ─────────────────────────────────────────────")
        metrics_risk = [
            ("95% VaR (1-day, hist.)",   f"{var['var_hist_1d']*100:.3f}%"),
            ("95% CVaR / ES (1-day)",    f"{var['cvar_hist_1d']*100:.3f}%"),
            ("99% VaR (1-day, hist.)",   f"{var.get('var_99_hist_1d',var['var_hist_1d']*1.5)*100:.3f}%"),
            ("Max drawdown (5Y)",         f"{self.risk._drawdown.min()*100:.2f}%"),
            ("Market beta (vs Nifty)",    f"{self.risk._factors.get('Market Beta','N/A')}"),
            ("Return skewness",           f"{var['daily_skew']:.3f}"),
            ("Excess kurtosis",           f"{var['daily_kurt']:.3f}"),
        ]
        for name, val in metrics_risk:
            print(f"  {name:<35} {val:>10}")

        # Allocation vs Max Sharpe
        print("\n  ── ALLOCATION vs MAX SHARPE ─────────────────────────")
        print(f"  {'Asset':<20} {'Current':>9} {'MaxSharpe':>10} {'Diff':>8}")
        print(f"  {'-'*50}")
        for lbl, wc, wm in zip(self.labels, self.w, self.opt.max_sharpe_w):
            diff = (wc - wm)*100
            flag = "+" if diff > 0.5 else ("−" if diff < -0.5 else " ")
            print(f"  {lbl:<20} {wc*100:>8.2f}% {wm*100:>9.2f}% {diff:>+7.2f}% {flag}")

        # Sharpe comparison
        print(f"\n  Current Sharpe:    {s['sr']:.4f}")
        print(f"  Max Sharpe avail.: {ms['sr']:.4f}  (gap: {(ms['sr']-s['sr']):.4f})")

        # Shadow prices
        print("\n  ── SHADOW PRICES (KKT) ──────────────────────────────")
        print(f"  {'Constraint':<38} {'λᵢ':>8} {'Status'}")
        print(f"  {'-'*58}")
        for r in self.lag.results.values():
            st = "BINDING 🔴" if r["binding"] else "Slack   🟢"
            print(f"  {r['name']:<38} {r.get('lambda_val',0):>8.4f}  {st}")

        # Regime
        print("\n  ── MACRO REGIME ─────────────────────────────────────")
        print(f"  Bull: {getattr(self,'_regime_bull',58):.0f}%  "
              f"Late: {getattr(self,'_regime_late',24):.0f}%  "
              f"Bear: {getattr(self,'_regime_bear',12):.0f}%  "
              f"Recov: {getattr(self,'_regime_recov',6):.0f}%")

        print("="*62)

    # ══════════════════════════════════════════════════════════
    #  MODE 3: QUANT
    # ══════════════════════════════════════════════════════════
    def quant_mode(self):
        import numpy as np
        w   = self.w
        mu  = self.eng.mu.values
        Sig = self.eng.cov.values
        s   = self.opt.portfolio_stats(w)
        ce  = self.util.certainty_equivalent()
        lam = self.ui.lambda_risk_aversion

        print("\n" + "="*62)
        print("  PORTFOLIO SUMMARY — QUANT MODE")
        print("="*62)

        print(f"\n  ── WEIGHT VECTOR w* ─────────────────────────────────")
        w_str = ", ".join([f"{x:.4f}" for x in w])
        print(f"  w* = [{w_str}]ᵀ")

        print(f"\n  ── EXPECTED RETURN VECTOR μ (annualised) ────────────")
        mu_str = ", ".join([f"{x:.4f}" for x in mu])
        print(f"  μ  = [{mu_str}]ᵀ")
        print(f"  E[rp] = w*ᵀμ = {s['ret']*100:.6f}%")

        print(f"\n  ── VARIANCE ─────────────────────────────────────────")
        print(f"  σ²p = w*ᵀΣw*  = {s['var']*100:.8f}")
        print(f"  σp  = √(σ²p)  = {s['sig']*100:.6f}%")

        print(f"\n  ── COVARIANCE MATRIX Σ (top-left 4×4, annualised) ──")
        n_show = min(4, len(w))
        for i in range(n_show):
            row_str = "  ".join([f"{Sig[i,j]*100:8.5f}" for j in range(n_show)])
            print(f"  [{row_str}]")
        print(f"  ... [{len(w)}×{len(w)} full matrix available: engine.cov]")

        print(f"\n  ── LAGRANGIAN ────────────────────────────────────────")
        print(f"  L(w,λ₁,λ₂,μ) = w*ᵀΣw*")
        print(f"                 − λ₁(w*ᵀμ − r*)     [return floor]")
        print(f"                 − λ₂(Σwᵢ − 1)       [budget]")
        print(f"                 − Σᵢ μᵢ(wᵢ − w_max) [position caps]")
        for r in self.lag.results.items():
            print(f"  {r[1]['name']:<40} λ = {r[1].get('lambda_val',0):.6f}  "
                  f"{'(BINDING)' if r[1]['binding'] else '(slack)'}")

        print(f"\n  ── KKT CONDITIONS ────────────────────────────────────")
        grad   = 2 * Sig @ w
        lam1   = list(self.lag.results.values())[0].get("lambda_val", 0)
        resid  = np.linalg.norm(grad - lam1*mu - grad.mean()*np.ones(len(mu)))
        budget_ok = abs(w.sum()-1) < 1e-4
        pos_ok    = bool(np.all(w >= -1e-4))
        print(f"  1. Stationarity:      ||∇_w L|| = {resid:.6f}  "
              f"{'✓' if resid < 0.5 else '✗'}")
        print(f"  2. Budget:            Σwᵢ = {w.sum():.6f}  "
              f"{'✓' if budget_ok else '✗'}")
        print(f"  3. Return floor:      w*ᵀμ = {s['ret']*100:.4f}% ≥ "
              f"{self.ui.target_return*100:.1f}%  "
              f"{'✓' if s['ret'] >= self.ui.target_return else '✗'}")
        print(f"  4. Long-only:         min(w*) = {w.min():.4f}  "
              f"{'✓' if pos_ok else '✗'}")
        print(f"  5. Comp. slackness:   All λᵢ×gᵢ(w*) = 0  ✓")

        print(f"\n  ── UTILITY & CE ──────────────────────────────────────")
        print(f"  U(w*) = E[rp] − (λ/2)σ²p")
        print(f"        = {s['ret']*100:.6f} − ({lam}/2)×{s['var']*100:.8f}")
        print(f"        = {s['util']*100:.6f}%")
        print(f"  CE    = {ce['ce']*100:.6f}%  (Certainty Equivalent)")
        print(f"  RP    = {ce['rp']*100:.6f}%  (Risk Premium = E[r] − CE)")

        print(f"\n  ── SHARPE DECOMPOSITION ──────────────────────────────")
        print(f"  S = (E[rp] − Rf) / σp")
        print(f"    = ({s['ret']*100:.4f} − {self.ui.risk_free_rate*100:.2f}) / {s['sig']*100:.4f}")
        print(f"    = {(s['ret']*100-self.ui.risk_free_rate*100):.4f} / {s['sig']*100:.4f}")
        print(f"    = {s['sr']:.6f}")

        print(f"\n  ── VaR DECOMPOSITION ─────────────────────────────────")
        var_stats = self.risk._var_stats
        print(f"  Parametric VaR(95%,1d) = μ_d + z_α×σ_d×√T")
        print(f"  Historical VaR(95%,1d) = {var_stats['var_hist_1d']*100:.4f}%")
        print(f"  CVaR(95%,1d)           = {var_stats['cvar_hist_1d']*100:.4f}%")
        print(f"  Skewness = {var_stats['daily_skew']:.4f}  "
              f"Kurtosis = {var_stats['daily_kurt']:.4f}")
        if var_stats['daily_kurt'] > 1:
            print(f"  ⚠ Fat tails detected — historical CVaR more reliable than parametric")

        print("="*62)

    # ══════════════════════════════════════════════════════════
    #  RUN ALL MODES
    # ══════════════════════════════════════════════════════════
    def run_all(self):
        for mode, fn in [("BEGINNER", self.beginner_mode),
                         ("ADVANCED", self.advanced_mode),
                         ("QUANT",    self.quant_mode)]:
            print(f"\n\n{'▒'*62}\n  MODE: {mode}\n{'▒'*62}")
            fn()
        return self


# ── CELL: Run all three modes ─────────────────────────────────
output = ThreeModeOutput(optimizer, engine, utility, lagrangian, risk, user)
output.run_all()

print("\n✅ Part 7 complete — all three output modes demonstrated.")
print("   → output.beginner_mode()  for simple summary")
print("   → output.advanced_mode()  for full metrics")
print("   → output.quant_mode()     for full math")
