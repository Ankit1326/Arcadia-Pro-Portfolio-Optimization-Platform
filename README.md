# 🚀 Arcadia Pro – AI Multi-Agent Portfolio Optimization Platform

> **An Institutional-Grade Portfolio Optimization & Risk Analytics Platform powered by AI, LangGraph, and Modern Portfolio Theory.**

Arcadia Pro is an end-to-end quantitative portfolio management system that combines classical finance, mathematical optimization, multi-agent reasoning, and AI-generated investment insights into a single interactive platform.

Designed as a Bloomberg Terminal-inspired research platform, it allows users to optimize portfolios, analyze portfolio risk, perform scenario analysis, and receive explainable AI recommendations.

---

# 🌟 Features

## 📊 Portfolio Optimization

- Markowitz Mean-Variance Optimization
- Maximum Sharpe Portfolio
- Minimum Variance Portfolio
- Utility-Optimal Portfolio
- Efficient Frontier Generation
- Black-Litterman Return Adjustment
- Risk Aversion Based Optimization
- Long Only & Custom Constraint Support

---

## 📈 Risk Analytics

- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Conditional VaR (Expected Shortfall)
- Rolling Sharpe Ratio
- Maximum Drawdown
- Drawdown Duration
- Volatility Analysis
- Portfolio Correlation Matrix
- Diversification Metrics

---

## 🧮 Mathematical Finance

Implemented using academic portfolio management techniques including:

- Modern Portfolio Theory (Markowitz)
- Utility Theory
- Certainty Equivalent
- Risk Premium
- Lagrangian Optimization
- Karush-Kuhn-Tucker (KKT) Conditions
- Shadow Price Analysis
- Efficient Frontier
- Convex Optimization

---

## 🤖 AI-Powered Investment Insights

The application integrates Claude AI to generate intelligent portfolio analysis in three modes:

### Beginner Mode

- Simple language
- Portfolio health
- Easy-to-understand recommendations

### Advanced Mode

- Professional portfolio report
- Risk statistics
- Performance metrics
- Constraint analysis

### Quant Mode

- Matrix calculations
- Utility function evaluation
- Lagrangian formulation
- KKT verification
- Shadow prices
- Mathematical derivations

---

# 🧠 LangGraph Multi-Agent Architecture

The platform includes a custom LangGraph workflow consisting of multiple intelligent agents.

### Agent 1

Portfolio Optimization Agent

Responsibilities

- Mean-Variance Optimization
- Black-Litterman
- Efficient Frontier
- Sharpe Optimization

---

### Agent 2

Risk & Constraint Agent

Responsibilities

- Lagrangian Optimization
- Constraint Verification
- KKT Conditions
- Shadow Prices
- Utility Evaluation

---

### Agent 3

Debate & Decision Agent

Responsibilities

- Compare outputs of Agent 1 & Agent 2
- Validate optimization quality
- Generate final investment recommendation
- Quality assurance

---

# 🏗 Architecture

```
                    User Input
                         │
                         ▼
              Data Collection Layer
                         │
                         ▼
             Return & Risk Engine
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
  Portfolio Optimization         Risk Analytics
          │                             │
          ▼                             ▼
     Agent 1                     Agent 2
          └──────────────┬──────────────┘
                         ▼
                 Debate Engine
                         ▼
                  Claude AI Analysis
                         ▼
               Interactive Dashboard
```

---

# 📂 Project Structure

```
Arcadia-Pro/

│
├── README.md
│
├── PART1
│     Data Collection
│
├── PART2
│     Markowitz Optimizer
│
├── PART3
│     Utility Theory
│
├── PART4
│     Risk Analytics
│
├── PART5
│     AI Insights
│
├── PART6
│     Interactive Dashboard
│
├── PART7
│     Beginner / Advanced / Quant Modes
│
├── PART8
│     Master Notebook
│
├── langgraph_full_pipeline.py
│
├── MATH_REFERENCE.md
│
└── assets/
```

---

# 📚 Mathematical Models Used

The project implements

- Markowitz Portfolio Optimization
- Black-Litterman Model
- CAPM
- Utility Theory
- CARA Utility
- CRRA Utility
- Efficient Frontier
- Sharpe Ratio
- VaR
- CVaR
- Hidden Markov Models
- Lagrangian Optimization
- KKT Conditions
- Shadow Pricing

---

# 💻 Technology Stack

## Programming

- Python 3.11+

## Libraries

- NumPy
- Pandas
- SciPy
- Plotly
- CVXPY
- yfinance
- LangGraph
- Anthropic Claude SDK

---

# 🌐 Deployment

The project can run on

- ✅ Google Colab
- ✅ Jupyter Notebook
- ✅ VS Code
- ✅ Replit
- ✅ Local Python Environment

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Arcadia-Pro.git

cd Arcadia-Pro
```

Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy pandas scipy plotly yfinance cvxpy anthropic langgraph
```

---

# 🔑 API Configuration

Create a `.env` file

```text
ANTHROPIC_API_KEY=your_api_key
```

or directly configure inside the notebook

```python
client = anthropic.Anthropic(
    api_key="YOUR_API_KEY"
)
```

---

# ▶ Running the Project

### Google Colab

Open

```
PART8_Master_Notebook.py
```

Run all cells.

---

### Replit

1. Import the repository into Replit.

2. Add your Anthropic API key to Secrets.

```
ANTHROPIC_API_KEY
```

3. Click **Run**.

The application will launch directly inside Replit.

---

# 📊 Dashboard Features

Interactive dashboard includes

- Portfolio Allocation
- Efficient Frontier
- Risk Metrics
- Correlation Heatmap
- Portfolio Performance
- Drawdown Curve
- Rolling Sharpe
- AI Investment Report

---

# 🎯 Key Highlights

- Multi-Agent AI Architecture
- Institutional Portfolio Optimization
- Explainable AI
- Academic Finance Models
- Interactive Visualizations
- Constraint Optimization
- Production-ready Python Code
- Beginner to Quant-level Reporting

---

# 📖 Future Improvements

- Live Broker Integration
- Alpaca Trading API
- Zerodha Kite Integration
- Real-Time Market Data
- PostgreSQL Storage
- User Authentication
- Portfolio Backtesting Engine
- Reinforcement Learning Allocation
- Multi-Objective Optimization
- Streamlit Cloud Deployment

---

# 📸 Application Preview

The following screenshots demonstrate the core functionality of Arcadia Pro.

| Dashboard | 
|-----------|
| ![Dashboard](screenshots/dashboard.png) | 

| Risk Analytics | 
|----------------|
| ![Risk Analytics](screenshots/risk.png) | 

| Portfolio Allocation | 
|----------------------|
| ![Allocation](screenshots/allocation.png) | 

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.

2. Create a feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Added new feature"
```

4. Push to your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Ankit Singh**

B.Tech, Metallurgical & Materials Engineering  
MANIT Bhopal

Interests

- Quantitative Finance
- Portfolio Optimization
- Machine Learning
- Financial Engineering
- Multi-Agent AI Systems
- Risk Analytics

---

## ⭐ If you found this project useful, consider giving it a Star!
