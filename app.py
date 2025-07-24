"""
Markov Chain Trading Signal Generator - Complete Fixed Version
Focus: Pure HMM Regime Detection + User-Controlled Position Sizing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Markov Chain Trading Signals",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .signal-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 3px solid;
    }
    
    .buy-signal {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-color: #28a745;
        color: #155724;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-color: #dc3545;
        color: #721c24;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-color: #ffc107;
        color: #856404;
    }
    
    .regime-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    
    .strength-meter {
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HMMSignalGenerator:
    """Simplified HMM focused on regime detection and signal quality - FIXED VERSION"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.regime_colors = {0: '#dc3545', 1: '#ffc107', 2: '#28a745'}
        self.regime_icons = {0: '📉', 1: '➡️', 2: '📈'}
    
    def prepare_features(self, data):
        """Enhanced feature engineering for regime detection - FIXED"""
        # Ensure we have enough data
        if len(data) < 60:
            raise ValueError("Need at least 60 days of data for reliable analysis")
        
        # Calculate features step by step to avoid alignment issues
        returns = data['Close'].pct_change()
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volume features with proper handling
        volume_sma = data['Volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = data['Volume'] / volume_sma
        
        # Price momentum
        momentum_5 = data['Close'].pct_change(5)
        momentum_20 = data['Close'].pct_change(20)
        
        # Volatility (using returns)
        volatility = returns.rolling(window=20, min_periods=1).std()
        
        # RSI
        rsi = self.calculate_rsi(data['Close'], 14)
        
        # Create features DataFrame with explicit alignment
        features = pd.DataFrame({
            'returns': returns,
            'log_returns': log_returns,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'momentum_5': momentum_5,
            'momentum_20': momentum_20,
            'rsi': rsi
        }, index=data.index)
        
        # Fill any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Drop first 30 rows to ensure clean data
        features_clean = features.iloc[30:].copy()
        
        if len(features_clean) < 50:
            raise ValueError("Insufficient clean data after feature calculation")
        
        return features_clean
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index - FIXED"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use exponential moving average for smoother calculation
        avg_gain = gain.ewm(span=period, min_periods=1).mean()
        avg_loss = loss.ewm(span=period, min_periods=1).mean()
        
        # Avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def fit_model(self, features):
        """Fit HMM model to features - FIXED"""
        try:
            # Ensure no infinite or NaN values
            features_array = features.values
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_array)
            
            # Fit Gaussian Mixture Model
            self.model = GaussianMixture(
                n_components=self.n_states,
                covariance_type='full',
                random_state=42,
                max_iter=100,
                init_params='kmeans'
            )
            
            self.model.fit(scaled_features)
            
            # Predict states and probabilities
            states = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)
            
            # Sort regimes by average return (Bear=0, Sideways=1, Bull=2)
            regime_returns = {}
            for regime in range(self.n_states):
                regime_mask = states == regime
                if np.sum(regime_mask) > 0:
                    regime_returns[regime] = features.loc[features.index[regime_mask], 'returns'].mean()
                else:
                    regime_returns[regime] = 0
            
            # Create mapping to ensure Bear=0, Bull=2, Sideways=1
            sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
            regime_mapping = {sorted_regimes[i][0]: i for i in range(len(sorted_regimes))}
            
            # Remap states
            remapped_states = np.array([regime_mapping[state] for state in states])
            
            # Remap probabilities
            remapped_probs = np.zeros_like(probabilities)
            for old_regime, new_regime in regime_mapping.items():
                remapped_probs[:, new_regime] = probabilities[:, old_regime]
            
            return remapped_states, remapped_probs
            
        except Exception as e:
            raise ValueError(f"Model fitting failed: {str(e)}")
    
    def analyze_regimes(self, data, features, states, probabilities):
        """Comprehensive regime analysis - FIXED"""
        regime_stats = {}
        
        # Align data with features (both should have same length after cleaning)
        aligned_data = data.loc[features.index]
        
        for regime in range(self.n_states):
            regime_mask = states == regime
            regime_indices = features.index[regime_mask]
            
            if len(regime_indices) > 0:
                regime_features = features.loc[regime_indices]
                
                regime_stats[regime] = {
                    'name': self.regime_names[regime],
                    'icon': self.regime_icons[regime],
                    'color': self.regime_colors[regime],
                    'days': len(regime_indices),
                    'percentage': len(regime_indices) / len(features) * 100,
                    'avg_return': regime_features['returns'].mean() * 100,
                    'volatility': regime_features['returns'].std() * 100,
                    'avg_volume_ratio': regime_features['volume_ratio'].mean(),
                    'persistence': self.calculate_persistence(states, regime)
                }
        
        return regime_stats
    
    def calculate_persistence(self, states, regime):
        """Calculate regime persistence probability - FIXED"""
        if len(states) < 2:
            return 0.0
        
        same_regime_transitions = 0
        total_regime_days = 0
        
        for i in range(1, len(states)):
            if states[i-1] == regime:
                total_regime_days += 1
                if states[i] == regime:
                    same_regime_transitions += 1
        
        return (same_regime_transitions / total_regime_days * 100) if total_regime_days > 0 else 0.0
    
    def generate_signal(self, current_regime, confidence, regime_stats):
        """Generate trading signal based on current regime - FIXED"""
        regime_name = self.regime_names[current_regime]
        
        # Signal logic
        if confidence >= 0.7:  # High confidence threshold
            if regime_name == 'Bull':
                signal = 'BUY'
                signal_strength = min(10, max(6, int(confidence * 12)))
            elif regime_name == 'Bear':
                signal = 'SELL'
                signal_strength = min(10, max(6, int(confidence * 12)))
            else:  # Sideways
                signal = 'HOLD'
                signal_strength = max(3, int(confidence * 8))
        else:
            signal = 'HOLD'
            signal_strength = max(1, int(confidence * 6))
        
        return {
            'signal': signal,
            'strength': signal_strength,
            'regime': regime_name,
            'confidence': confidence * 100,
            'regime_stats': regime_stats.get(current_regime, {})
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Markov Chain Trading Signals</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="regime-card">
    <strong>🎯 Pure HMM Signal Generation</strong><br>
    Advanced regime detection using Hidden Markov Models. You control position sizing and risk management 
    based on high-quality market regime signals.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("📊 Analysis Settings")
    
    symbol = st.sidebar.text_input(
        "Stock Symbol:",
        value="SOFI",
        help="Enter any valid stock ticker"
    ).upper()
    
    lookback_days = st.sidebar.slider(
        "Analysis Period (Days):",
        min_value=100,
        max_value=800,
        value=252,
        help="Number of trading days to analyze"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Signal Confidence Threshold:",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Minimum confidence required for BUY/SELL signals"
    )
    
    # Analysis button
    if st.sidebar.button("🚀 Generate Signal", type="primary"):
        try:
            with st.spinner(f"Analyzing {symbol} market regimes..."):
                # Download data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(lookback_days * 1.8))
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error(f"❌ No data available for {symbol}")
                    return
                
                # Ensure we have enough data
                if len(data) < 100:
                    st.error(f"❌ Insufficient data for {symbol}. Need at least 100 days.")
                    return
                
                # Initialize HMM
                hmm = HMMSignalGenerator()
                
                # Prepare features
                features = hmm.prepare_features(data)
                
                # Fit model and get predictions
                states, probabilities = hmm.fit_model(features)
                
                # Get current state
                current_regime = states[-1]
                current_confidence = probabilities[-1].max()
                
                # Analyze regimes
                regime_stats = hmm.analyze_regimes(data, features, states, probabilities)
                
                # Generate signal
                signal_data = hmm.generate_signal(current_regime, current_confidence, regime_stats)
                
                # Display results  
                display_results(symbol, signal_data, regime_stats, data, features, states, probabilities)
                
        except Exception as e:
            st.error(f"❌ Error analyzing {symbol}: {str(e)}")
            st.info("💡 Try using a different symbol (like AAPL) or reduce the analysis period.")

def display_results(symbol, signal_data, regime_stats, data, features, states, probabilities):
    """Display comprehensive analysis results"""
    
    # Main signal display
    signal = signal_data['signal']
    strength = signal_data['strength']
    regime = signal_data['regime']
    confidence = signal_data['confidence']
    
    # Signal box styling
    if signal == 'BUY':
        signal_class = 'buy-signal'
        signal_icon = '🟢'
    elif signal == 'SELL':
        signal_class = 'sell-signal'
        signal_icon = '🔴'
    else:
        signal_class = 'hold-signal'
        signal_icon = '🟡'
    
    st.markdown(f"""
    <div class="signal-box {signal_class}">
        <h2>{signal_icon} {signal} SIGNAL</h2>
        <h3>{regime} Market Regime</h3>
        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        <div class="strength-meter">
            Signal Strength: {'█' * strength}{'░' * (10-strength)} ({strength}/10)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current regime details
    current_regime_stats = signal_data['regime_stats']
    if current_regime_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Daily Return",
                f"{current_regime_stats.get('avg_return', 0):.2f}%",
                help="Average daily return in current regime"
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{current_regime_stats.get('volatility', 0):.1f}%",
                help="Daily volatility in current regime"
            )
        
        with col3:
            st.metric(
                "Persistence",
                f"{current_regime_stats.get('persistence', 0):.1f}%",
                help="Probability regime continues tomorrow"
            )
        
        with col4:
            st.metric(
                "Days in Regime",
                f"{current_regime_stats.get('days', 0)}",
                help="Total days in this regime historically"
            )
    
    # Regime statistics
    st.header("📊 Regime Analysis")
    
    regime_data = []
    for regime_id, stats in regime_stats.items():
        regime_data.append({
            'Regime': f"{stats['icon']} {stats['name']}",
            'Days': stats['days'],
            'Percentage': f"{stats['percentage']:.1f}%",
            'Avg Return': f"{stats['avg_return']:.2f}%",
            'Volatility': f"{stats['volatility']:.1f}%",
            'Persistence': f"{stats['persistence']:.1f}%"
        })
    
    regime_df = pd.DataFrame(regime_data)
    st.dataframe(regime_df, use_container_width=True)
    
    # Position sizing guidance
    st.header("💡 Position Sizing Guidance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="regime-card">
        <h4>🎯 Suggested Position Sizes</h4>
        <ul>
        <li><strong>Conservative:</strong> 2-5% of portfolio</li>
        <li><strong>Moderate:</strong> 5-10% of portfolio</li>
        <li><strong>Aggressive:</strong> 10-20% of portfolio</li>
        </ul>
        <p><em>Adjust based on your risk tolerance and the signal strength above.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        persistence_rate = current_regime_stats.get('persistence', 0) if current_regime_stats else 0
        st.markdown(f"""
        <div class="regime-card">
        <h4>⚠️ Risk Management</h4>
        <ul>
        <li><strong>Stop Loss:</strong> Set based on your pain threshold</li>
        <li><strong>Take Profit:</strong> Consider regime persistence rate</li>
        <li><strong>Review:</strong> Check weekly for regime changes</li>
        <li><strong>Current Persistence:</strong> {persistence_rate:.1f}%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Price chart with regime overlay
    st.header("📈 Price Chart with Regime Detection")
    
    # Create price chart with regime overlay
    try:
        # Get aligned data for chart
        chart_data = data.loc[features.index]
        chart_states = states
        
        # Limit to recent data for better visualization
        recent_data = chart_data.tail(min(200, len(chart_data)))
        recent_states = chart_states[-len(recent_data):]
        
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='black', width=2)
        ))
        
        # Add regime background colors
        regime_colors = {0: 'rgba(220, 53, 69, 0.2)', 1: 'rgba(255, 193, 7, 0.2)', 2: 'rgba(40, 167, 69, 0.2)'}
        
        if len(recent_states) > 0:
            current_regime = recent_states[0]
            regime_start = 0
            
            for i in range(1, len(recent_states)):
                if recent_states[i] != current_regime or i == len(recent_states) - 1:
                    end_idx = i - 1 if i < len(recent_states) else i - 1
                    
                    fig.add_shape(
                        type="rect",
                        x0=recent_data.index[regime_start],
                        x1=recent_data.index[end_idx],
                        y0=recent_data['Close'].min() * 0.95,
                        y1=recent_data['Close'].max() * 1.05,
                        fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.2)'),
                        layer="below",
                        line_width=0,
                    )
                    
                    regime_start = i
                    current_regime = recent_states[i] if i < len(recent_states) else current_regime
        
        fig.update_layout(
            title=f"{symbol} Price with Market Regimes",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Chart display error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
    <p>🔮 Analysis completed for {symbol} • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} • 
    <strong>Remember:</strong> You control position sizing and risk management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
