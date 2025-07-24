"""
Markov Chain Trading Signal Generator - Dimension Issues Fixed
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
</style>
""", unsafe_allow_html=True)

class HMMSignalGenerator:
    """Simplified HMM focused on regime detection - ALL DIMENSION ISSUES FIXED"""
    
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.regime_colors = {0: '#dc3545', 1: '#ffc107', 2: '#28a745'}
        self.regime_icons = {0: '📉', 1: '➡️', 2: '📈'}
    
    def prepare_features(self, data):
        """Simplified feature engineering - DIMENSION FIXED"""
        # Ensure we have enough data
        if len(data) < 60:
            raise ValueError("Need at least 60 days of data for reliable analysis")
        
        # Basic price features - ensure all are Series
        close_prices = data['Close'].copy()
        volume = data['Volume'].copy()
        
        # Calculate returns
        returns = close_prices.pct_change()
        
        # Simple volatility
        volatility = returns.rolling(window=20, min_periods=1).std()
        
        # Volume ratio
        volume_ma = volume.rolling(window=20, min_periods=1).mean()
        volume_ratio = volume / volume_ma
        
        # Momentum
        momentum = close_prices.pct_change(10)
        
        # Simple RSI
        rsi = self.calculate_rsi_simple(close_prices)
        
        # Create clean features DataFrame
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'rsi': rsi
        }, index=data.index)
        
        # Remove NaN values and clean data
        features = features.dropna()
        
        # Additional cleaning
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if len(features) < 50:
            raise ValueError("Insufficient clean data after processing")
        
        return features
    
    def calculate_rsi_simple(self, prices):
        """Simple RSI calculation - DIMENSION ISSUES FIXED"""
        # Ensure we're working with a Series
        if hasattr(prices, 'values'):
            price_series = prices.copy()
        else:
            price_series = pd.Series(prices)
        
        # Calculate price changes
        delta = price_series.diff()
        
        # Get gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Simple moving averages
        avg_gains = gains.rolling(window=14, min_periods=1).mean()
        avg_losses = losses.rolling(window=14, min_periods=1).mean()
        
        # RSI calculation
        rs = avg_gains / (avg_losses + 0.0001)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Fill any remaining NaN
        rsi = rsi.fillna(50)
        
        return rsi
    
    def fit_model(self, features):
        """Fit HMM model - DIMENSION ISSUES FIXED"""
        try:
            # Convert to numpy array safely
            features_values = features.values.astype(float)
            
            # Handle any remaining problematic values
            features_values = np.nan_to_num(features_values, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_values)
            
            # Fit model
            self.model = GaussianMixture(
                n_components=self.n_states,
                covariance_type='diag',  # Simpler covariance for stability
                random_state=42,
                max_iter=50
            )
            
            self.model.fit(scaled_features)
            
            # Get predictions
            states = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)
            
            # Sort regimes by return (Bear=0, Sideways=1, Bull=2)
            regime_returns = []
            for regime in range(self.n_states):
                mask = states == regime
                if np.sum(mask) > 0:
                    avg_return = features.iloc[mask]['returns'].mean()
                    regime_returns.append((regime, avg_return))
                else:
                    regime_returns.append((regime, 0))
            
            # Sort by returns
            regime_returns.sort(key=lambda x: x[1])
            
            # Create mapping
            regime_mapping = {}
            for new_regime, (old_regime, _) in enumerate(regime_returns):
                regime_mapping[old_regime] = new_regime
            
            # Remap states
            remapped_states = np.array([regime_mapping[state] for state in states])
            
            # Remap probabilities
            remapped_probs = np.zeros_like(probabilities)
            for old_regime, new_regime in regime_mapping.items():
                remapped_probs[:, new_regime] = probabilities[:, old_regime]
            
            return remapped_states, remapped_probs
            
        except Exception as e:
            raise ValueError(f"Model fitting failed: {str(e)}")
    
    def analyze_regimes(self, features, states):
        """Analyze regime characteristics - SIMPLIFIED"""
        regime_stats = {}
        
        for regime in range(self.n_states):
            mask = states == regime
            if np.sum(mask) > 0:
                regime_features = features.iloc[mask]
                
                regime_stats[regime] = {
                    'name': self.regime_names[regime],
                    'icon': self.regime_icons[regime],
                    'color': self.regime_colors[regime],
                    'days': int(np.sum(mask)),
                    'percentage': float(np.sum(mask) / len(states) * 100),
                    'avg_return': float(regime_features['returns'].mean() * 100),
                    'volatility': float(regime_features['returns'].std() * 100),
                    'persistence': self.calculate_persistence(states, regime)
                }
        
        return regime_stats
    
    def calculate_persistence(self, states, regime):
        """Calculate persistence rate"""
        if len(states) < 2:
            return 0.0
        
        same_transitions = 0
        total_regime_days = 0
        
        for i in range(1, len(states)):
            if states[i-1] == regime:
                total_regime_days += 1
                if states[i] == regime:
                    same_transitions += 1
        
        return (same_transitions / total_regime_days * 100) if total_regime_days > 0 else 0.0
    
    def generate_signal(self, current_regime, confidence, regime_stats):
        """Generate trading signal"""
        regime_name = self.regime_names[current_regime]
        
        if confidence >= 0.7:
            if regime_name == 'Bull':
                signal = 'BUY'
                strength = min(10, max(6, int(confidence * 12)))
            elif regime_name == 'Bear':
                signal = 'SELL'  
                strength = min(10, max(6, int(confidence * 12)))
            else:
                signal = 'HOLD'
                strength = max(3, int(confidence * 8))
        else:
            signal = 'HOLD'
            strength = max(1, int(confidence * 6))
        
        return {
            'signal': signal,
            'strength': strength,
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
    Advanced regime detection using Hidden Markov Models. You control position sizing and risk management.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    symbol = st.sidebar.text_input("Stock Symbol:", value="SOFI").upper()
    lookback_days = st.sidebar.slider("Analysis Period (Days):", 100, 500, 200)
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.5, 0.9, 0.7, 0.05)
    
    if st.sidebar.button("🚀 Generate Signal", type="primary"):
        try:
            with st.spinner(f"Analyzing {symbol}..."):
                # Download data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(lookback_days * 1.5))
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data.empty or len(data) < 100:
                    st.error(f"❌ Insufficient data for {symbol}")
                    return
                
                # Initialize and run analysis
                hmm = HMMSignalGenerator()
                features = hmm.prepare_features(data)
                states, probabilities = hmm.fit_model(features)
                
                # Current state
                current_regime = states[-1]
                current_confidence = probabilities[-1].max()
                
                # Analyze regimes
                regime_stats = hmm.analyze_regimes(features, states)
                
                # Generate signal
                signal_data = hmm.generate_signal(current_regime, current_confidence, regime_stats)
                
                # Display results
                display_results(symbol, signal_data, regime_stats, data, features, states)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try AAPL or reduce the analysis period")

def display_results(symbol, signal_data, regime_stats, data, features, states):
    """Display results - SIMPLIFIED"""
    
    signal = signal_data['signal']
    strength = signal_data['strength']
    regime = signal_data['regime']
    confidence = signal_data['confidence']
    
    # Signal display
    signal_colors = {'BUY': 'buy-signal', 'SELL': 'sell-signal', 'HOLD': 'hold-signal'}
    signal_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
    
    st.markdown(f"""
    <div class="signal-box {signal_colors[signal]}">
        <h2>{signal_icons[signal]} {signal} SIGNAL</h2>
        <h3>{regime} Market Regime</h3>
        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        <div class="strength-meter">
            Signal Strength: {'█' * strength}{'░' * (10-strength)} ({strength}/10)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current regime metrics
    current_stats = signal_data['regime_stats']
    if current_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Return", f"{current_stats.get('avg_return', 0):.2f}%")
        with col2:
            st.metric("Volatility", f"{current_stats.get('volatility', 0):.1f}%")
        with col3:
            st.metric("Persistence", f"{current_stats.get('persistence', 0):.1f}%")
        with col4:
            st.metric("Days", f"{current_stats.get('days', 0)}")
    
    # Regime table
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
    
    st.dataframe(pd.DataFrame(regime_data), use_container_width=True)
    
    # Position guidance
    st.header("💡 Position Sizing Guidance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="regime-card">
        <h4>🎯 Suggested Sizes</h4>
        <ul>
        <li><strong>Conservative:</strong> 2-5%</li>
        <li><strong>Moderate:</strong> 5-10%</li>
        <li><strong>Aggressive:</strong> 10-20%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="regime-card">
        <h4>⚠️ Risk Management</h4>
        <ul>
        <li><strong>Stop Loss:</strong> Your choice</li>
        <li><strong>Take Profit:</strong> Based on persistence</li>
        <li><strong>Review:</strong> Weekly checks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple chart
    st.header("📈 Price Chart")
    try:
        chart_data = data.loc[features.index].tail(100)
        chart_states = states[-len(chart_data):]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Trend",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Chart display temporarily unavailable")

if __name__ == "__main__":
    main()
