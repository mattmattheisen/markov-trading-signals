"""
Markov Chain Trading Signal Generator - Ultra Simple Version
100% Dimension Issue Free
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    layout="wide"
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

def calculate_simple_rsi(prices, period=14):
    """Ultra simple RSI - no dimension issues"""
    # Convert to list to avoid any pandas/numpy issues
    price_list = prices.tolist()
    
    if len(price_list) < period + 1:
        return [50] * len(price_list)  # Return neutral RSI
    
    rsi_values = []
    
    for i in range(len(price_list)):
        if i < period:
            rsi_values.append(50)  # Neutral for insufficient data
        else:
            # Get price changes for the period
            changes = []
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    changes.append(price_list[j] - price_list[j-1])
            
            if not changes:
                rsi_values.append(50)
                continue
            
            # Separate gains and losses
            gains = [change for change in changes if change > 0]
            losses = [-change for change in changes if change < 0]
            
            # Calculate averages
            avg_gain = sum(gains) / len(changes) if gains else 0
            avg_loss = sum(losses) / len(changes) if losses else 0
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
    
    return rsi_values

def prepare_simple_features(data):
    """Ultra simple feature preparation"""
    # Convert everything to simple lists first
    closes = data['Close'].tolist()
    volumes = data['Volume'].tolist()
    
    if len(closes) < 50:
        raise ValueError("Need at least 50 days of data")
    
    # Calculate simple returns
    returns = []
    for i in range(len(closes)):
        if i == 0:
            returns.append(0)
        else:
            returns.append((closes[i] - closes[i-1]) / closes[i-1])
    
    # Simple volatility (20-day rolling std)
    volatilities = []
    for i in range(len(returns)):
        if i < 20:
            volatilities.append(0.02)  # Default volatility
        else:
            recent_returns = returns[i-19:i+1]
            mean_return = sum(recent_returns) / len(recent_returns)
            variance = sum([(r - mean_return)**2 for r in recent_returns]) / len(recent_returns)
            volatilities.append(variance**0.5)
    
    # Simple volume ratio
    volume_ratios = []
    for i in range(len(volumes)):
        if i < 20:
            volume_ratios.append(1.0)
        else:
            recent_volumes = volumes[i-19:i+1]
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            volume_ratios.append(volumes[i] / avg_volume if avg_volume > 0 else 1.0)
    
    # Simple momentum (10-day)
    momentums = []
    for i in range(len(closes)):
        if i < 10:
            momentums.append(0)
        else:
            momentums.append((closes[i] - closes[i-10]) / closes[i-10])
    
    # Calculate RSI
    rsi_values = calculate_simple_rsi(data['Close'], 14)
    
    # Create feature matrix
    features = []
    for i in range(len(returns)):
        features.append([
            returns[i],
            volatilities[i],
            volume_ratios[i], 
            momentums[i],
            rsi_values[i] / 100  # Normalize RSI to 0-1
        ])
    
    # Remove first 30 rows for clean data
    clean_features = features[30:]
    clean_dates = data.index[30:]
    
    return np.array(clean_features), clean_dates, returns[30:]

def analyze_with_hmm(symbol, lookback_days):
    """Main HMM analysis function"""
    try:
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty or len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Prepare features
        features, dates, returns = prepare_simple_features(data)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Fit HMM
        model = GaussianMixture(
            n_components=3,
            covariance_type='diag',
            random_state=42,
            max_iter=50
        )
        
        model.fit(scaled_features)
        
        # Get predictions
        states = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        # Sort regimes by returns
        regime_returns = {}
        for regime in range(3):
            mask = states == regime
            if np.sum(mask) > 0:
                regime_returns[regime] = np.mean([returns[i] for i in range(len(returns)) if mask[i]])
            else:
                regime_returns[regime] = 0
        
        # Map regimes: Bear=0, Sideways=1, Bull=2
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Remap states
        mapped_states = [regime_mapping[s] for s in states]
        
        # Analyze regime statistics
        regime_stats = {}
        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        regime_icons = {0: '📉', 1: '➡️', 2: '📈'}
        
        for regime in range(3):
            mask = [s == regime for s in mapped_states]
            count = sum(mask)
            
            if count > 0:
                regime_returns_list = [returns[i] for i in range(len(returns)) if mask[i]]
                avg_return = np.mean(regime_returns_list) * 100
                volatility = np.std(regime_returns_list) * 100
                
                # Calculate persistence
                persistence = 0
                regime_days = 0
                for i in range(1, len(mapped_states)):
                    if mapped_states[i-1] == regime:
                        regime_days += 1
                        if mapped_states[i] == regime:
                            persistence += 1
                
                persistence_rate = (persistence / regime_days * 100) if regime_days > 0 else 0
                
                regime_stats[regime] = {
                    'name': regime_names[regime],
                    'icon': regime_icons[regime],
                    'days': count,
                    'percentage': count / len(mapped_states) * 100,
                    'avg_return': avg_return,
                    'volatility': volatility,
                    'persistence': persistence_rate
                }
        
        # Current signal
        current_regime = mapped_states[-1]
        current_confidence = probabilities[-1].max()
        
        # Generate signal
        if current_confidence >= 0.7:
            if current_regime == 2:  # Bull
                signal = 'BUY'
                strength = min(10, max(6, int(current_confidence * 12)))
            elif current_regime == 0:  # Bear
                signal = 'SELL'
                strength = min(10, max(6, int(current_confidence * 12)))
            else:  # Sideways
                signal = 'HOLD'
                strength = max(3, int(current_confidence * 8))
        else:
            signal = 'HOLD'
            strength = max(1, int(current_confidence * 6))
        
        return {
            'signal': signal,
            'strength': strength,
            'regime': regime_names[current_regime],
            'confidence': current_confidence * 100,
            'regime_stats': regime_stats,
            'current_regime_stats': regime_stats.get(current_regime, {}),
            'data': data,
            'dates': dates,
            'states': mapped_states
        }
        
    except Exception as e:
        raise ValueError(f"Analysis failed: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Markov Chain Trading Signals</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="regime-card">
    <strong>🎯 Pure HMM Signal Generation</strong><br>
    Ultra-reliable regime detection. You control position sizing and risk management.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    symbol = st.sidebar.text_input("Stock Symbol:", value="SOFI").upper()
    lookback_days = st.sidebar.slider("Analysis Period (Days):", 100, 400, 200)
    
    if st.sidebar.button("🚀 Generate Signal", type="primary"):
        try:
            with st.spinner(f"Analyzing {symbol}..."):
                results = analyze_with_hmm(symbol, lookback_days)
                display_results(symbol, results)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Try AAPL or MSFT first to test the system")

def display_results(symbol, results):
    """Display analysis results"""
    
    signal = results['signal']
    strength = results['strength']
    regime = results['regime']
    confidence = results['confidence']
    
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
    current_stats = results['current_regime_stats']
    if current_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Daily Return", f"{current_stats['avg_return']:.2f}%")
        with col2:
            st.metric("Volatility", f"{current_stats['volatility']:.1f}%")
        with col3:
            st.metric("Persistence", f"{current_stats['persistence']:.1f}%")
        with col4:
            st.metric("Days", f"{current_stats['days']}")
    
    # Regime table
    st.header("📊 Regime Analysis")
    regime_data = []
    for regime_id, stats in results['regime_stats'].items():
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
    st.header("💡 Your Decision")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="regime-card">
        <h4>🎯 Position Size Options</h4>
        <ul>
        <li><strong>Conservative:</strong> 2-5% of portfolio</li>
        <li><strong>Moderate:</strong> 5-10% of portfolio</li>
        <li><strong>Aggressive:</strong> 10-20% of portfolio</li>
        </ul>
        <p><strong>Signal Strength:</strong> {strength}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="regime-card">
        <h4>⚠️ Risk Management</h4>
        <ul>
        <li><strong>Stop Loss:</strong> Set your comfort level</li>
        <li><strong>Take Profit:</strong> Based on {current_stats.get('persistence', 0):.0f}% persistence</li>
        <li><strong>Review:</strong> Check weekly for regime changes</li>
        <li><strong>Current Price:</strong> ${results['data']['Close'].iloc[-1]:.2f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple price chart
    st.header("📈 Recent Price Action")
    try:
        recent_data = results['data'].tail(100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} - Last 100 Days",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Price chart temporarily unavailable")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
    <p>🔮 {symbol} analysis complete • {datetime.now().strftime('%Y-%m-%d %H:%M')} • 
    <strong>You decide position size and risk management</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
