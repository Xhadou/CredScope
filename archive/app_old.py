"""Streamlit Dashboard for CredScope Credit Risk Assessment

Interactive web interface for loan officers and analysts to evaluate applicants.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.credscope.api.predictor import CreditRiskPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CredScope - Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .approve {
        color: #28a745;
        font-weight: bold;
    }
    .reject {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the predictor (cached)"""
    try:
        predictor = CreditRiskPredictor(models_dir="models")
        predictor.load_models()
        return predictor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None


def create_gauge_chart(value, title):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "lightblue"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(features_data):
    """Create horizontal bar chart for feature importance"""
    df = pd.DataFrame(features_data)

    # Color based on impact
    colors = ['red' if impact == 'increases' else 'green'
              for impact in df['impact']]

    fig = go.Figure(go.Bar(
        x=df['shap_value'],
        y=df['feature'],
        orientation='h',
        marker_color=colors,
        text=df['shap_value'].round(3),
        textposition='auto',
    ))

    fig.update_layout(
        title="Top Features Influencing Decision",
        xaxis_title="SHAP Value (Impact on Default Probability)",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )

    return fig


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">💳 CredScope</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Alternative Credit Risk Assessment Platform</p>', unsafe_allow_html=True)

    # Load predictor
    predictor = load_predictor()

    if predictor is None:
        st.error("⚠️ Models could not be loaded. Please ensure model files are in the 'models' directory.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for rejection (higher = stricter)"
        )

        st.markdown("---")

        st.header("📊 Model Status")
        st.success("✅ Models Loaded")
        st.info(f"🔢 Features: {len(predictor.feature_names) if predictor.feature_names else 'Unknown'}")

        st.markdown("---")

        st.header("ℹ️ About")
        st.markdown("""
        **CredScope** uses alternative data sources to evaluate credit risk:
        - Payment behavior patterns
        - Bureau credit history
        - Employment & income stability
        - Document completeness
        - Behavioral indicators

        This enables fair assessment of applicants with thin credit files.
        """)

    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions", "📈 Analytics"])

    # TAB 1: Single Prediction
    with tab1:
        st.header("Evaluate Single Applicant")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Basic Information")

            amt_income = st.number_input(
                "Total Income ($)",
                min_value=0,
                value=180000,
                step=10000,
                help="Annual income of applicant"
            )

            amt_credit = st.number_input(
                "Loan Amount ($)",
                min_value=0,
                value=500000,
                step=10000,
                help="Credit amount requested"
            )

            amt_annuity = st.number_input(
                "Loan Annuity ($)",
                min_value=0,
                value=25000,
                step=1000,
                help="Annual payment amount"
            )

            amt_goods_price = st.number_input(
                "Goods Price ($)",
                min_value=0,
                value=450000,
                step=10000,
                help="Price of goods being financed"
            )

        with col2:
            st.subheader("👤 Demographic Information")

            age_years = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=41,
                help="Applicant's age"
            )
            days_birth = -int(age_years * 365.25)

            employment_years = st.number_input(
                "Employment Duration (years)",
                min_value=0,
                max_value=50,
                value=8,
                help="Years with current employer"
            )
            days_employed = -int(employment_years * 365.25)

            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=1
            )
            code_gender = 1 if gender == "Male" else 0

        st.subheader("🎯 External Credit Scores")
        col3, col4, col5 = st.columns(3)

        with col3:
            ext_source_1 = st.slider(
                "External Source 1",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.01,
                help="External credit score 1"
            )

        with col4:
            ext_source_2 = st.slider(
                "External Source 2",
                min_value=0.0,
                max_value=1.0,
                value=0.72,
                step=0.01,
                help="External credit score 2"
            )

        with col5:
            ext_source_3 = st.slider(
                "External Source 3",
                min_value=0.0,
                max_value=1.0,
                value=0.58,
                step=0.01,
                help="External credit score 3"
            )

        st.markdown("---")

        # Predict button
        if st.button("🔍 Evaluate Applicant", type="primary", use_container_width=True):
            with st.spinner("Analyzing application..."):
                # Prepare input
                applicant_data = {
                    'AMT_INCOME_TOTAL': amt_income,
                    'AMT_CREDIT': amt_credit,
                    'AMT_ANNUITY': amt_annuity,
                    'AMT_GOODS_PRICE': amt_goods_price,
                    'DAYS_BIRTH': days_birth,
                    'DAYS_EMPLOYED': days_employed,
                    'CODE_GENDER': code_gender,
                    'EXT_SOURCE_1': ext_source_1,
                    'EXT_SOURCE_2': ext_source_2,
                    'EXT_SOURCE_3': ext_source_3,
                }

                try:
                    # Get explanation
                    result = predictor.explain_prediction(applicant_data, top_n=10)

                    # Display results
                    st.markdown("---")
                    st.header("📊 Assessment Results")

                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        decision_color = "approve" if result['decision'] == "APPROVE" else "reject"
                        st.markdown(f'<div class="metric-card"><h3 class="{decision_color}">{result["decision"]}</h3><p>Decision</p></div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown(f'<div class="metric-card"><h3>{result["default_probability"]:.1%}</h3><p>Default Probability</p></div>', unsafe_allow_html=True)

                    with col3:
                        st.markdown(f'<div class="metric-card"><h3>{result["risk_level"]}</h3><p>Risk Level</p></div>', unsafe_allow_html=True)

                    with col4:
                        st.markdown(f'<div class="metric-card"><h3>{result["confidence"]:.1%}</h3><p>Confidence</p></div>', unsafe_allow_html=True)

                    # Gauge chart
                    st.plotly_chart(
                        create_gauge_chart(result['default_probability'], "Default Risk"),
                        use_container_width=True
                    )

                    # Feature importance
                    st.subheader("🔍 Key Factors Influencing Decision")
                    if 'top_features' in result:
                        st.plotly_chart(
                            create_feature_importance_chart(result['top_features']),
                            use_container_width=True
                        )

                        # Feature details table
                        with st.expander("📋 Detailed Feature Analysis"):
                            df_features = pd.DataFrame(result['top_features'])
                            df_features['shap_value'] = df_features['shap_value'].round(4)
                            df_features['feature_value'] = df_features['feature_value'].round(4)
                            st.dataframe(df_features, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)

    # TAB 2: Batch Predictions
    with tab2:
        st.header("Batch Application Processing")

        st.info("📁 Upload a CSV file with multiple applicant records for batch scoring.")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain columns: AMT_INCOME_TOTAL, AMT_CREDIT, DAYS_BIRTH, etc."
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} applicants")

                st.subheader("Preview Data")
                st.dataframe(df.head(10), use_container_width=True)

                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner("Processing applications..."):
                        # Make predictions
                        predictions, probabilities = predictor.predict(df, threshold=threshold)

                        # Add results to dataframe
                        df['default_probability'] = probabilities
                        df['predicted_class'] = predictions
                        df['decision'] = df['predicted_class'].map({0: 'APPROVE', 1: 'REJECT'})

                        # Display summary
                        st.header("📊 Batch Results")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Applications", len(df))
                        with col2:
                            approved = (df['decision'] == 'APPROVE').sum()
                            st.metric("Approved", approved, delta=f"{approved/len(df)*100:.1f}%")
                        with col3:
                            rejected = (df['decision'] == 'REJECT').sum()
                            st.metric("Rejected", rejected, delta=f"{rejected/len(df)*100:.1f}%")
                        with col4:
                            avg_prob = df['default_probability'].mean()
                            st.metric("Avg Default Prob", f"{avg_prob:.1%}")

                        # Distribution chart
                        fig = px.histogram(
                            df,
                            x='default_probability',
                            color='decision',
                            nbins=50,
                            title="Distribution of Default Probabilities",
                            labels={'default_probability': 'Default Probability', 'count': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)

                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Results",
                            data=csv,
                            file_name="credscope_batch_results.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # TAB 3: Analytics
    with tab3:
        st.header("📈 Model Analytics & Insights")

        st.info("This section provides insights into model performance and feature importance.")

        # Load feature importance
        importance_path = Path("models/feature_importance.csv")
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)

            st.subheader("🎯 Top 20 Most Important Features")

            fig = px.bar(
                importance_df.head(20),
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance Ranking",
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Feature categories
            st.subheader("📊 Feature Categories")

            # Categorize features
            def categorize_feature(name):
                if name.startswith('INT_'):
                    return 'Interaction'
                elif name.startswith('EXT_'):
                    return 'External Scores'
                elif name.startswith('BUREAU_'):
                    return 'Bureau Data'
                elif name.startswith('INST_'):
                    return 'Installments'
                elif name.startswith('CC_'):
                    return 'Credit Card'
                elif name.startswith('POS_'):
                    return 'POS/Cash'
                elif name.startswith('PREV_'):
                    return 'Previous Apps'
                elif name.startswith('DOC_'):
                    return 'Documents'
                else:
                    return 'Application'

            importance_df['category'] = importance_df['feature'].apply(categorize_feature)

            category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)

            fig = px.pie(
                values=category_importance.values,
                names=category_importance.index,
                title="Feature Importance by Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Feature importance file not found. Train the model first.")

        # Model information
        st.subheader("ℹ️ Model Information")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Ensemble Architecture:**
            - LightGBM (35.9%)
            - XGBoost (32.1%)
            - CatBoost (32.0%)
            - Meta-learner: Logistic Regression

            **Performance:**
            - AUC-ROC: 0.7908
            - Features: 522
            """)

        with col2:
            st.markdown("""
            **Key Feature Categories:**
            - External credit scores
            - Bureau credit history
            - Installment payments
            - Credit card usage
            - POS/Cash loans
            - Previous applications
            - Documents submitted
            - Interaction features
            """)


if __name__ == "__main__":
    main()
