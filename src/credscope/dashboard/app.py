"""Streamlit Dashboard for CredScope Credit Risk Assessment

Interactive web interface for loan officers and analysts to evaluate applicants.
Enhanced with professional styling and comprehensive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Color scheme - Modern professional palette
COLORS = {
    'primary': '#6366F1',       # Indigo
    'secondary': '#8B5CF6',     # Purple
    'success': '#10B981',       # Emerald
    'warning': '#F59E0B',       # Amber
    'danger': '#EF4444',        # Red
    'info': '#3B82F6',          # Blue
    'dark': '#1F2937',          # Gray-800
    'light': '#F3F4F6',         # Gray-100
    'background': '#0F172A',    # Slate-900
    'card': '#1E293B',          # Slate-800
    'card_hover': '#334155',    # Slate-700
    'text': '#E2E8F0',          # Slate-200
    'muted': '#94A3B8',         # Slate-400
    'border': '#334155',        # Slate-700
    'accent': '#818CF8',        # Indigo-400
}

# Risk level colors - Updated for new thresholds
RISK_COLORS = {
    'VERY_LOW': '#10B981',      # Green - safe
    'LOW': '#34D399',           # Light green - approve zone
    'MEDIUM': '#FBBF24',        # Yellow - review zone start
    'HIGH': '#F97316',          # Orange - review zone end  
    'VERY_HIGH': '#EF4444'      # Red - reject zone
}

# Page configuration
st.set_page_config(
    page_title="CredScope - Credit Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main Header */
    .main-header {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    
    .sub-header {{
        text-align: center;
        color: {COLORS['muted']};
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(145deg, {COLORS['card']} 0%, #252d3d 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    
    .metric-label {{
        color: {COLORS['muted']};
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Decision Badges - Modern pill style with animations */
    .decision-approve {{
        background: linear-gradient(135deg, {COLORS['success']} 0%, #059669 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 3rem;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        animation: pulse-green 2s infinite;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .decision-reject {{
        background: linear-gradient(135deg, {COLORS['danger']} 0%, #DC2626 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 3rem;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .decision-review {{
        background: linear-gradient(135deg, {COLORS['warning']} 0%, #D97706 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 3rem;
        font-weight: 700;
        font-size: 1.4rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4);
        animation: pulse-yellow 2s infinite;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    @keyframes pulse-green {{
        0%, 100% {{ box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4); }}
        50% {{ box-shadow: 0 8px 35px rgba(16, 185, 129, 0.6); }}
    }}
    
    @keyframes pulse-yellow {{
        0%, 100% {{ box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4); }}
        50% {{ box-shadow: 0 8px 35px rgba(245, 158, 11, 0.6); }}
    }}
    
    /* Risk Level Badges */
    .risk-very-low {{ color: {RISK_COLORS['VERY_LOW']}; }}
    .risk-low {{ color: {RISK_COLORS['LOW']}; }}
    .risk-medium {{ color: {RISK_COLORS['MEDIUM']}; }}
    .risk-high {{ color: {RISK_COLORS['HIGH']}; }}
    .risk-very-high {{ color: {RISK_COLORS['VERY_HIGH']}; }}
    
    /* Section Headers */
    .section-header {{
        color: {COLORS['text']};
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['primary']};
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS['card']};
        padding: 0.5rem;
        border-radius: 1rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 0.5rem;
        color: {COLORS['muted']};
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['card']} 0%, {COLORS['background']} 100%);
    }}
    
    [data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}
    
    /* Button Styling */
    .stButton > button {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }}
    
    /* Dataframe Styling */
    .stDataFrame {{
        border-radius: 1rem;
        overflow: hidden;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background-color: {COLORS['card']};
        border-radius: 0.5rem;
    }}
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {{
        border-radius: 0.75rem;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['secondary']};
    }}
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


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create an enhanced gauge chart for risk visualization"""
    
    # Determine color based on new thresholds: <20% approve, 20-50% review, >50% reject
    if value < 0.10:
        bar_color = RISK_COLORS['VERY_LOW']
    elif value < 0.20:
        bar_color = RISK_COLORS['LOW']
    elif value < 0.35:
        bar_color = RISK_COLORS['MEDIUM']
    elif value < 0.50:
        bar_color = RISK_COLORS['HIGH']
    else:
        bar_color = RISK_COLORS['VERY_HIGH']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': COLORS['text']}},
        title={'text': title, 'font': {'size': 18, 'color': COLORS['muted']}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': COLORS['muted'],
                'tickfont': {'color': COLORS['muted']},
                'tickvals': [0, 20, 50, 100],
                'ticktext': ['0%', '20%', '50%', '100%']
            },
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': COLORS['card'],
            'borderwidth': 2,
            'bordercolor': COLORS['card'],
            'steps': [
                {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.3)'},    # Approve zone - green
                {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.3)'},   # Review zone - yellow
                {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.3)'}    # Reject zone - red
            ],
            'threshold': {
                'line': {'color': '#FFFFFF', 'width': 4},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=50, b=30),
        font={'color': COLORS['text']}
    )
    
    return fig


def create_feature_importance_chart(features_data: list) -> go.Figure:
    """Create enhanced horizontal bar chart for feature importance"""
    df = pd.DataFrame(features_data)
    
    # Sort by absolute SHAP value
    df = df.reindex(df['shap_value'].abs().sort_values(ascending=True).index)
    
    # Color based on impact direction
    colors = [RISK_COLORS['VERY_HIGH'] if impact == 'increases' else RISK_COLORS['VERY_LOW']
              for impact in df['impact']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['shap_value'],
        y=df['feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{v:+.3f}" for v in df['shap_value']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11),
        hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<br>Value: %{customdata:.4f}<extra></extra>",
        customdata=df['feature_value']
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['muted'], line_width=1)
    
    fig.update_layout(
        title=dict(
            text="🔍 Key Factors Influencing Decision",
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title="SHAP Value (Impact on Default Probability)",
            titlefont=dict(color=COLORS['muted']),
            tickfont=dict(color=COLORS['muted']),
            gridcolor='rgba(148, 163, 184, 0.1)',
            zerolinecolor=COLORS['muted']
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=COLORS['text'], size=11),
            gridcolor='rgba(148, 163, 184, 0.1)'
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=200, r=80, t=60, b=60),
        showlegend=False
    )
    
    return fig


def create_score_distribution_chart(probabilities: np.ndarray, decisions: np.ndarray = None) -> go.Figure:
    """Create score distribution histogram with color overlay"""
    
    fig = go.Figure()
    
    if decisions is not None:
        # Separate by decision
        approved_probs = probabilities[decisions == 0]
        rejected_probs = probabilities[decisions == 1]
        
        fig.add_trace(go.Histogram(
            x=approved_probs,
            name='Approved',
            marker_color=COLORS['success'],
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=rejected_probs,
            name='Rejected',
            marker_color=COLORS['danger'],
            opacity=0.7,
            nbinsx=30
        ))
    else:
        fig.add_trace(go.Histogram(
            x=probabilities,
            name='All Applications',
            marker=dict(
                color=COLORS['primary'],
            ),
            opacity=0.8,
            nbinsx=40
        ))
    
    # Add threshold line
    fig.add_vline(x=0.5, line_dash="dash", line_color=COLORS['text'], line_width=2,
                  annotation_text="Threshold", annotation_position="top")
    
    fig.update_layout(
        title=dict(
            text="📊 Score Distribution",
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title="Default Probability",
            titlefont=dict(color=COLORS['muted']),
            tickfont=dict(color=COLORS['muted']),
            gridcolor='rgba(148, 163, 184, 0.1)',
        ),
        yaxis=dict(
            title="Count",
            titlefont=dict(color=COLORS['muted']),
            tickfont=dict(color=COLORS['muted']),
            gridcolor='rgba(148, 163, 184, 0.1)',
        ),
        barmode='overlay',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS['text'])
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def create_model_comparison_chart() -> go.Figure:
    """Create model performance comparison chart"""
    
    models = ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble']
    aucs = [0.7900, 0.7895, 0.7889, 0.7908]
    weights = [35.9, 32.1, 32.0, 100]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model AUC-ROC Scores', 'Ensemble Weights'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # AUC bar chart
    colors = [COLORS['info'], COLORS['warning'], COLORS['success'], COLORS['primary']]
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=aucs,
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=[f"{auc:.4f}" for auc in aucs],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=12),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Ensemble weights pie chart
    fig.add_trace(
        go.Pie(
            labels=models[:3],
            values=weights[:3],
            marker=dict(colors=colors[:3]),
            textinfo='label+percent',
            textfont=dict(color='white'),
            hole=0.4,
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text']),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update axes
    fig.update_xaxes(tickfont=dict(color=COLORS['text']), gridcolor='rgba(148, 163, 184, 0.1)', row=1, col=1)
    fig.update_yaxes(tickfont=dict(color=COLORS['muted']), gridcolor='rgba(148, 163, 184, 0.1)', 
                     range=[0.78, 0.80], row=1, col=1)
    
    return fig


def create_feature_category_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Create feature category importance breakdown"""
    
    def categorize_feature(name):
        if name.startswith('INT_'):
            return 'Interactions'
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
    
    importance_df = importance_df.copy()
    importance_df['category'] = importance_df['feature'].apply(categorize_feature)
    category_stats = importance_df.groupby('category').agg({
        'importance': ['sum', 'count', 'mean']
    }).round(2)
    category_stats.columns = ['total_importance', 'feature_count', 'avg_importance']
    category_stats = category_stats.sort_values('total_importance', ascending=True)
    
    fig = go.Figure()
    
    # Color mapping
    category_colors = {
        'Interactions': COLORS['primary'],
        'External Scores': COLORS['secondary'],
        'Bureau Data': COLORS['info'],
        'Installments': COLORS['success'],
        'Credit Card': COLORS['warning'],
        'POS/Cash': '#EC4899',  # Pink
        'Previous Apps': '#14B8A6',  # Teal
        'Documents': '#8B5CF6',  # Purple
        'Application': COLORS['muted']
    }
    
    colors = [category_colors.get(cat, COLORS['muted']) for cat in category_stats.index]
    
    fig.add_trace(go.Bar(
        x=category_stats['total_importance'],
        y=category_stats.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=[f"{v:,.0f} ({c} features)" for v, c in zip(category_stats['total_importance'], category_stats['feature_count'])],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11),
    ))
    
    fig.update_layout(
        title=dict(
            text="📊 Feature Importance by Category",
            font=dict(size=18, color=COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title="Total Importance Score",
            titlefont=dict(color=COLORS['muted']),
            tickfont=dict(color=COLORS['muted']),
            gridcolor='rgba(148, 163, 184, 0.1)',
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=COLORS['text'], size=12),
        ),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=120, t=60, b=60),
        showlegend=False
    )
    
    return fig


def render_metric_card(value: str, label: str, color: str = None):
    """Render a styled metric card"""
    color_style = f"color: {color};" if color else ""
    return f'''
    <div class="metric-card">
        <div class="metric-value" style="{color_style}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    '''


def render_decision_badge(decision: str):
    """Render a styled decision badge"""
    icons = {'APPROVE': '✅', 'REVIEW': '⚠️', 'REJECT': '❌'}
    icon = icons.get(decision, '📋')
    class_name = f"decision-{decision.lower()}"
    return f'<span class="{class_name}">{icon} {decision}</span>'


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🛡️ CredScope</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Alternative Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("⚠️ Models could not be loaded. Please ensure model files are in the 'models' directory.")
        st.info("Required files: `lightgbm_model.txt`, `xgboost_model.json`, `catboost_model.cbm`")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <p style="color: #E2E8F0; font-size: 0.85rem; margin: 0;">
                <strong>Decision Thresholds:</strong><br>
                ✅ <span style="color: #10B981;">Approve</span>: &lt; 20% risk<br>
                ⚠️ <span style="color: #F59E0B;">Review</span>: 20-50% risk<br>
                ❌ <span style="color: #EF4444;">Reject</span>: &gt; 50% risk
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        threshold = 0.5  # Fixed threshold based on new logic
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<span style='color: {COLORS['success']}'>● Online</span>", unsafe_allow_html=True)
        with col2:
            st.caption(f"{len(predictor.feature_names) if predictor.feature_names else '?'} features")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Stats")
        st.metric("AUC-ROC", "0.7908", "Production")
        st.metric("Models", "3", "Ensemble")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        <small>
        <b>CredScope</b> uses alternative data to evaluate credit risk fairly:
        <ul>
            <li>Payment behavior patterns</li>
            <li>Bureau credit history</li>
            <li>Employment stability</li>
            <li>Document completeness</li>
        </ul>
        Enabling fair assessment for thin-file applicants.
        </small>
        """, unsafe_allow_html=True)
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Processing", "📈 Model Analytics"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.markdown('<div class="section-header">Evaluate Single Applicant</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 💰 Financial Information")
            
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
            st.markdown("##### 👤 Personal Information")
            
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
        
        st.markdown("##### 📈 External Credit Scores")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            ext_source_1 = st.slider(
                "External Score 1",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.01,
                help="External credit score 1"
            )
        
        with col4:
            ext_source_2 = st.slider(
                "External Score 2",
                min_value=0.0,
                max_value=1.0,
                value=0.72,
                step=0.01,
                help="External credit score 2"
            )
        
        with col5:
            ext_source_3 = st.slider(
                "External Score 3",
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
                    # Get prediction with explanation
                    result = predictor.explain_prediction(applicant_data, top_n=10)
                    
                    # Results section
                    st.markdown("---")
                    st.markdown('<div class="section-header">📊 Assessment Results</div>', unsafe_allow_html=True)
                    
                    # Main metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        decision = result['decision']
                        decision_html = render_decision_badge(decision)
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            {decision_html}
                            <div class="metric-label" style="margin-top: 0.5rem;">Decision</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        prob = result['default_probability']
                        prob_color = RISK_COLORS['VERY_HIGH'] if prob > 0.6 else RISK_COLORS['MEDIUM'] if prob > 0.4 else RISK_COLORS['VERY_LOW']
                        st.markdown(render_metric_card(f"{prob:.1%}", "Default Probability", prob_color), unsafe_allow_html=True)
                    
                    with col3:
                        risk = result['risk_level']
                        risk_color = RISK_COLORS.get(risk, COLORS['text'])
                        st.markdown(render_metric_card(risk.replace('_', ' '), "Risk Level", risk_color), unsafe_allow_html=True)
                    
                    with col4:
                        conf = result['confidence']
                        st.markdown(render_metric_card(f"{conf:.1%}", "Confidence", COLORS['info']), unsafe_allow_html=True)
                    
                    # Gauge chart
                    st.markdown("")
                    col_gauge, col_info = st.columns([2, 1])
                    
                    with col_gauge:
                        st.plotly_chart(
                            create_gauge_chart(result['default_probability'], "Default Risk Score"),
                            use_container_width=True
                        )
                    
                    with col_info:
                        # Determine zone description based on probability
                        if prob < 0.20:
                            zone_text = "This applicant falls in the <b style='color: #10B981;'>AUTO-APPROVE</b> zone."
                            zone_desc = "Low default risk - recommended for automatic approval."
                        elif prob < 0.50:
                            zone_text = "This applicant falls in the <b style='color: #F59E0B;'>MANUAL REVIEW</b> zone."
                            zone_desc = "Moderate risk - requires manual review by a loan officer."
                        else:
                            zone_text = "This applicant falls in the <b style='color: #EF4444;'>HIGH RISK</b> zone."
                            zone_desc = "High default risk - recommended for rejection or additional scrutiny."
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: {COLORS['text']}; margin-bottom: 1rem;">📋 Risk Assessment</h4>
                            <p style="color: {COLORS['muted']}; font-size: 0.9rem;">
                                Based on analysis of <b style="color: {COLORS['primary']}">{len(predictor.feature_names) if predictor.feature_names else 500}+</b> features,
                                this applicant has a <b style="color: {prob_color}">{prob:.1%}</b> probability of default.
                            </p>
                            <p style="color: {COLORS['text']}; font-size: 0.9rem; margin-top: 1rem;">
                                {zone_text}
                            </p>
                            <p style="color: {COLORS['muted']}; font-size: 0.85rem; margin-top: 0.5rem; font-style: italic;">
                                {zone_desc}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Feature importance
                    if 'top_features' in result and result['top_features']:
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
        st.markdown('<div class="section-header">Batch Application Processing</div>', unsafe_allow_html=True)
        
        st.info("📁 Upload a CSV file with multiple applicant records for batch scoring.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain columns: AMT_INCOME_TOTAL, AMT_CREDIT, DAYS_BIRTH, etc."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df):,} applicants")
                
                st.markdown("##### Preview Data")
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
                        st.markdown('<div class="section-header">📊 Batch Results</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(render_metric_card(f"{len(df):,}", "Total Applications"), unsafe_allow_html=True)
                        with col2:
                            approved = (df['decision'] == 'APPROVE').sum()
                            st.markdown(render_metric_card(f"{approved:,}", "Approved", COLORS['success']), unsafe_allow_html=True)
                        with col3:
                            rejected = (df['decision'] == 'REJECT').sum()
                            st.markdown(render_metric_card(f"{rejected:,}", "Rejected", COLORS['danger']), unsafe_allow_html=True)
                        with col4:
                            avg_prob = df['default_probability'].mean()
                            st.markdown(render_metric_card(f"{avg_prob:.1%}", "Avg Default Prob"), unsafe_allow_html=True)
                        
                        # Score distribution chart
                        st.plotly_chart(
                            create_score_distribution_chart(
                                probabilities, 
                                predictions
                            ),
                            use_container_width=True
                        )
                        
                        # Results table
                        st.markdown("##### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download Results",
                            data=csv,
                            file_name="credscope_batch_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # TAB 3: Analytics
    with tab3:
        st.markdown('<div class="section-header">Model Analytics & Insights</div>', unsafe_allow_html=True)
        
        # Model comparison section
        st.markdown("##### 🏆 Model Performance Comparison")
        st.plotly_chart(create_model_comparison_chart(), use_container_width=True)
        
        # Feature importance section
        importance_path = Path("models/feature_importance.csv")
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🎯 Top 15 Most Important Features")
                
                # Create bar chart for top features
                top_features = importance_df.head(15).copy()
                top_features = top_features.sort_values('importance', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker=dict(
                        color=top_features['importance'],
                        colorscale=[[0, COLORS['info']], [1, COLORS['primary']]],
                        line=dict(color='rgba(255,255,255,0.2)', width=1)
                    ),
                    text=[f"{v:,.0f}" for v in top_features['importance']],
                    textposition='outside',
                    textfont=dict(color=COLORS['text'], size=10)
                ))
                
                fig.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title="Importance Score",
                        titlefont=dict(color=COLORS['muted']),
                        tickfont=dict(color=COLORS['muted']),
                        gridcolor='rgba(148, 163, 184, 0.1)',
                    ),
                    yaxis=dict(
                        title="",
                        tickfont=dict(color=COLORS['text'], size=10),
                    ),
                    margin=dict(l=180, r=80, t=20, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### 📊 Feature Categories")
                st.plotly_chart(create_feature_category_chart(importance_df), use_container_width=True)
        else:
            st.warning("Feature importance file not found. Train the model first.")
        
        # Model information cards
        st.markdown("##### ℹ️ Model Architecture")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {COLORS['info']};">🌲 LightGBM</h4>
                <p style="color: {COLORS['muted']};">
                    <b>Weight:</b> 35.9%<br>
                    <b>AUC:</b> 0.7900<br>
                    <b>Type:</b> Gradient Boosting
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {COLORS['warning']};">🚀 XGBoost</h4>
                <p style="color: {COLORS['muted']};">
                    <b>Weight:</b> 32.1%<br>
                    <b>AUC:</b> 0.7895<br>
                    <b>Type:</b> Gradient Boosting
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: {COLORS['success']};">🐱 CatBoost</h4>
                <p style="color: {COLORS['muted']};">
                    <b>Weight:</b> 32.0%<br>
                    <b>AUC:</b> 0.7889<br>
                    <b>Type:</b> Gradient Boosting
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance metrics summary
        st.markdown("##### 📈 Performance Summary")
        
        metrics_data = {
            'Metric': ['AUC-ROC', 'Gini Coefficient', 'Total Features', 'Data Sources'],
            'Value': ['0.7908', '0.5816', '522', '7 tables'],
            'Status': ['✅ Production', '✅ Good', '✅ Optimized', '✅ Complete']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
