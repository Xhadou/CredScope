# CredScope AI - Project Final Report (v3.0)

## 1. Executive Summary

**CredScope AI** is an advanced credit default risk prediction system designed to enhance loan approval decisions for financial institutions. By leveraging a sophisticated ensemble of machine learning models (LightGBM, XGBoost, CatBoost) and a modern full-stack architecture, the system delivers high-accuracy predictions with explainability and fairness at its core.

**Key Achievements:**
*   **Model Performance:** Achieved a ROC-AUC of **0.7908** on the test set, significantly outperforming the baseline (0.745).
*   **Business Impact:** Projected **$12.5M annual ROI** per $1B portfolio through reduced default rates (2.1% reduction) and optimized approval thresholds.
*   **Architecture:** Fully containerized microservices architecture with a FastAPI backend and a React/Vite frontend.
*   **Explainability:** Integrated SHAP (SHapley Additive exPlanations) for global and local interpretability, ensuring regulatory compliance and trust.
*   **Fairness:** rigorous fairness auditing across sensitive attributes (Age, Gender) with drift detection mechanisms in place.

---

## 2. System Architecture

The system follows a modern microservices pattern, separating the Machine Learning pipeline, the Serving API, and the User Interface.

### 2.1. Machine Learning Pipeline (`src/credscope`)
*   **Data Processing:** Robust ETL pipeline handling 7 relational tables (Home Credit dataset).
*   **Feature Engineering:** Generation of **522 features**, including domain-specific ratios (e.g., Credit-to-Income), aggregations, and interaction terms.
*   **Model Ensemble:** A weighted ensemble of three gradient boosting frameworks:
    *   **LightGBM:** High efficiency on large datasets.
    *   **XGBoost:** Robust regularization and performance.
    *   **CatBoost:** Superior handling of categorical variables.
*   **Optimization:** Hyperparameters tuned using **Optuna** for maximum predictive power.

### 2.2. Serving API (`src/credscope/api`)
*   **Framework:** **FastAPI** (Python 3.11) for high-performance asynchronous request handling.
*   **Endpoints:**
    *   `/predict`: Real-time default probability scoring.
    *   `/predict/explain`: SHAP-based feature contribution analysis.
    *   `/predict/batch`: Bulk processing for portfolio analysis.
*   **Validation:** Pydantic models ensure strict data validation for inputs and outputs.

### 2.3. Frontend Dashboard (`demo-site`)
*   **Framework:** **React 18** with **Vite** build tool.
*   **Styling:** **Tailwind CSS** for responsive, modern design.
*   **Components:**
    *   `RealTimePrediction.jsx`: Interactive form for loan officer usage.
    *   `EnsembleArchitecture.jsx`: Visual representation of the model weights and flow.
    *   `Hero.jsx`: Landing page with key value propositions.
*   **Visualization:** Recharts and Framer Motion for dynamic data presentation.

---

## 3. Methodology & Implementation

### 3.1. Data Strategy
The dataset consists of historical loan applications augmented with bureau data, previous applications, and transactional history.
*   **Handling Imbalance:** Addressed the 8% default rate using scale_pos_weight and stratified sampling.
*   **Missing Values:** Imputed using a combination of median strategies and specific flags for structural missingness.

### 3.2. Feature Engineering
Key engineered features that drove performance:
1.  `PAYMENT_RATE`: Annuity / Credit Amount (Strongest predictor).
2.  `EXT_SOURCE_MEAN`: Average of external credit scores.
3.  `CREDIT_TO_INCOME_RATIO`: Debt burden indicator.
4.  `DAYS_EMPLOYED_PERC`: Employment stability relative to age.

### 3.3. Model Training
*   **Training Strategy:** 5-Fold Stratified Cross-Validation to ensure stability.
*   **Ensemble Weights:**
    *   LightGBM: 0.40
    *   XGBoost: 0.35
    *   CatBoost: 0.25
*   **Calibration:** Probability calibration ensures the output scores represent true default probabilities.

---

## 4. Results & Performance

### 4.1. Predictive Metrics
| Metric | Baseline | Final Ensemble | Improvement |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.7450 | **0.7908** | +6.1% |
| **Gini Coefficient** | 0.4900 | **0.5816** | +18.7% |
| **Precision (Top 10%)** | 0.28 | **0.34** | +21.4% |

### 4.2. Business Impact Analysis
Using a standard cost-benefit matrix (Cost of False Negative = 10x Cost of False Positive):
*   **Optimal Threshold:** 0.085 (Probability of Default)
*   **Default Rate Reduction:** From 8.1% to 6.0%
*   **Approval Rate:** Maintained at 72% while filtering high-risk applicants.

### 4.3. Fairness & Drift
*   **Demographic Parity:** The model shows minimal bias across gender groups (Disparate Impact Ratio > 0.95).
*   **Drift Detection:** Automated KS-Test and PSI (Population Stability Index) monitoring implemented to trigger retraining if data distribution shifts.

---

## 5. Deployment & Operations

### 5.1. Containerization
*   **API Dockerfile:** Multi-stage build based on `python:3.11-slim`, optimized for size and security.
*   **Dashboard Dockerfile:** Nginx-based serving for the React static build.
*   **Orchestration:** `docker-compose.yml` defines the full stack services including networking and volume management.

### 5.2. Monitoring
*   **Prometheus/Grafana:** (Planned) Integration for real-time latency and throughput monitoring.
*   **Drift Monitoring:** Scheduled jobs to check feature drift using the `src/credscope/monitoring` module.

---

## 6. Future Work

1.  **Alternative Data:** Integrate telco and utility data to improve scoring for "thin-file" clients.
2.  **Graph Neural Networks:** Explore GNNs to leverage relationships between applicants and previous applications.
3.  **LLM Integration:** Use Large Language Models to generate natural language explanations for loan rejection reasons.
4.  **A/B Testing Framework:** Implement infrastructure for live champion/challenger model testing.

---
*Generated by CredScope AI Engineering Team*
