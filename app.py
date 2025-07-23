"""
Markov Chain Trading Signal Generator
===================================
Jim Simons-inspired quantitative trading app using:
- Hidden Markov Models for regime detection
- Kelly Criterion for position sizing
- Transition probability analysis
- Real-time market state detection

Educational use only - not investment advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Simplified HMM implementation
from sklearn.mixture import GaussianMixture
from scipy import stats

# ------------- CONFIGURATION -------------
DEFAULTS = {
    "lookback_days": 252,        # 1 year of trading days
    "n_states": 3,               # Bull, Bear, Sideways
    "min_probability": 0.6,      # Minimum confidence for signals
    "kelly_fraction": 0.25,      # Conservative Kelly sizing (25% of full Kelly)
    "max_position": 0.10,        # Maximum 10% of portfolio per position
    "transition_threshold": 0.7, # Threshold for regime change signals
}

STATE_NAMES = {
    0: "📈 Bull Market",
    1: "📉 Bear Market", 
    2: "➡️ Sideways Market"
}

STATE_COLORS = {
    0: "#00ff00",  # Green
    1: "#ff0000",  # Red
    2: "#ffaa00",  # Orange
}

# ------------- DATA CLASSES -------------
@dataclass
class MarketState:
    current_state: int
    state_name: str
    probability: float
    transition_probs: Dict[int, float]
    kelly_size: float
    confidence: float

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    current_state: int
    state_probability: float
    kelly_fraction: float
    position_size: float
    confidence: float
    reasoning: str
    timestamp: datetime

# ------------- MARKET DATA FUNCTIONS -------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Get historical market data with technical indicators"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate returns and features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Technical indicators
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['rsi'] = calculate_rsi(df['Close'])
        
        # Price momentum features
        df['price_momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['price_momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        return df.dropna()
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ------------- MARKOV MODEL IMPLEMENTATION -------------
class MarkovRegimeDetector:
    """Hidden Markov Model for market regime detection"""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self.states = None
        self.transition_matrix = None
        
    def fit(self, features: np.ndarray) -> None:
        """Fit the HMM model to market data"""
        try:
            # Use Gaussian Mixture Model as a simplified HMM
            self.model = GaussianMixture(
                n_components=self.n_states,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            
            self.model.fit(features)
            
            # Predict states for all data
            self.states = self.model.predict(features)
            
            # Calculate transition matrix
            self._calculate_transition_matrix()
            
        except Exception as e:
            st.error(f"Error fitting Markov model: {e}")
            
    def _calculate_transition_matrix(self) -> None:
        """Calculate state transition probabilities"""
        if self.states is None:
            return
            
        n = len(self.states)
        transitions = np.zeros((self.n_states, self.n_states))
        
        for i in range(1, n):
            prev_state = self.states[i-1]
            curr_state = self.states[i]
            transitions[prev_state, curr_state] += 1
        
        # Normalize to get probabilities
        row_sums = transitions.sum(axis=1)
        self.transition_matrix = transitions / row_sums[:, np.newaxis]
        
    def predict_state(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict current market state and confidence"""
        if self.model is None:
            return 0, 0.0
            
        try:
            # Get state probabilities
            probs = self.model.predict_proba(features.reshape(1, -1))[0]
            
            # Current state is the most likely one
            current_state = np.argmax(probs)
            confidence = probs[current_state]
            
            return current_state, confidence
            
        except Exception as e:
            st.error(f"Error predicting state: {e}")
            return 0, 0.0
    
    def get_transition_probabilities(self, current_state: int) -> Dict[int, float]:
        """Get transition probabilities from current state"""
        if self.transition_matrix is None:
            return {}
            
        return {
            i: self.transition_matrix[current_state, i] 
            for i in range(self.n_states)
        }

# ------------- KELLY CRITERION IMPLEMENTATION -------------
def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly Criterion fraction"""
    if avg_loss <= 0:
        return 0.0
        
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    
    # Return conservative fraction (never bet more than 25% of full Kelly)
    return max(0, min(kelly_fraction * 0.25, 0.10))

def calculate_position_size(kelly_fraction: float, portfolio_value: float, 
                          entry_price: float, stop_price: float, max_position: float = 0.10) -> int:
    """Calculate position size using Kelly Criterion"""
    if entry_price <= stop_price:
        return 0
        
    # Risk per share
    risk_per_share = entry_price - stop_price
    
    # Total dollars to risk based on Kelly
    dollars_to_risk = portfolio_value * kelly_fraction
    
    # Position size
    position_value = dollars_to_risk / (risk_per_share / entry_price)
    
    # Cap at maximum position size
    max_position_value = portfolio_value * max_position
    position_value = min(position_value, max_position_value)
    
    # Convert to shares
    shares = int(position_value / entry_price)
    
    return max(0, shares)

# ------------- SIGNAL GENERATION -------------
def generate_markov_signals(df: pd.DataFrame, detector: MarkovRegimeDetector, 
                           portfolio_value: float, cfg: dict) -> List[TradingSignal]:
    """Generate trading signals based on Markov regime detection"""
    signals = []
    
    if df.empty or detector.model is None:
        return signals
    
    try:
        # Prepare features for the latest data point
        latest = df.iloc[-1]
        features = np.array([
            latest['returns'],
            latest['volatility'],
            latest['volume_ratio'],
            latest['price_momentum_5'],
            latest['price_momentum_20']
        ])
        
        # Predict current state
        current_state, confidence = detector.predict_state(features)
        
        if confidence < cfg['min_probability']:
            return signals
        
        # Get transition probabilities
        transition_probs = detector.get_transition_probabilities(current_state)
        
        # Calculate historical performance for Kelly sizing
        # (Simplified - in production, you'd have more sophisticated backtesting)
        recent_returns = df['returns'].tail(60).dropna()
        win_rate = (recent_returns > 0).mean()
        avg_win = recent_returns[recent_returns > 0].mean() if (recent_returns > 0).any() else 0
        avg_loss = recent_returns[recent_returns < 0].mean() if (recent_returns < 0).any() else -0.01
        
        kelly_fraction = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Generate signal based on current state and transitions
        signal_type = "HOLD"
        reasoning = f"Current regime: {STATE_NAMES[current_state]} (Confidence: {confidence:.1%})"
        
        # Bull market regime
        if current_state == 0 and confidence > cfg['transition_threshold']:
            signal_type = "BUY"
            reasoning += f" | Strong bullish momentum detected"
            
        # Bear market regime  
        elif current_state == 1 and confidence > cfg['transition_threshold']:
            signal_type = "SELL"
            reasoning += f" | Strong bearish momentum detected"
            
        # Sideways market
        else:
            signal_type = "HOLD"
            reasoning += f" | Market in consolidation phase"
        
        # Calculate position size (assuming 2% stop loss for example)
        entry_price = latest['Close']
        stop_price = entry_price * 0.98  # 2% stop loss
        position_size = calculate_position_size(
            kelly_fraction, portfolio_value, entry_price, stop_price, cfg['max_position']
        )
        
        signal = TradingSignal(
            symbol=df.index.name or "UNKNOWN",
            signal_type=signal_type,
            current_state=current_state,
            state_probability=confidence,
            kelly_fraction=kelly_fraction,
            position_size=position_size,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        signals.append(signal)
        
    except Exception as e:
        st.error(f"Error generating signals: {e}")
    
    return signals

# ------------- VISUALIZATION -------------
def create_regime_chart(df: pd.DataFrame, detector: MarkovRegimeDetector) -> go.Figure:
    """Create chart showing price action with regime overlays"""
    if df.empty or detector.states is None:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Market Regimes', 'Returns', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart with regime coloring
    for state in range(detector.n_states):
        mask = detector.states == state
        state_dates = df.index[mask]
        state_prices = df['Close'][mask]
        
        if len(state_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=state_dates,
                    y=state_prices,
                    mode='markers',
                    marker=dict(color=STATE_COLORS[state], size=3),
                    name=STATE_NAMES[state],
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            line=dict(color='black', width=1),
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Returns chart
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['returns'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='Daily Returns',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue',
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title="Market Regime Analysis",
        height=800,
        xaxis_title="Date"
    )
    
    return fig

def create_transition_matrix_heatmap(detector: MarkovRegimeDetector) -> go.Figure:
    """Create heatmap of state transition probabilities"""
    if detector.transition_matrix is None:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=detector.transition_matrix,
        x=[STATE_NAMES[i] for i in range(detector.n_states)],
        y=[STATE_NAMES[i] for i in range(detector.n_states)],
        colorscale='RdYlGn',
        text=np.round(detector.transition_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Transition Probability")
    ))
    
    fig.update_layout(
        title="State Transition Probability Matrix",
        xaxis_title="To State",
        yaxis_title="From State",
        height=500
    )
    
    return fig

# ------------- STREAMLIT APP -------------
def main():
    st.set_page_config(
        page_title="Markov Trading Signals",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Markov Chain Trading Signal Generator")
    st.caption("Jim Simons-inspired quantitative trading using Hidden Markov Models & Kelly Criterion")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Model Configuration")
    
    # Model parameters
    with st.sidebar.expander("🧠 Markov Model Settings", expanded=True):
        n_states = st.selectbox("Number of Market States", [2, 3, 4], index=1)
        lookback_days = st.slider("Training Data (Days)", 60, 1000, DEFAULTS['lookback_days'])
        min_probability = st.slider("Minimum Signal Confidence", 0.5, 0.9, DEFAULTS['min_probability'])
    
    # Kelly Criterion settings
    with st.sidebar.expander("💰 Kelly Criterion Settings", expanded=True):
        portfolio_value = st.number_input("Portfolio Value ($)", 10000, 10000000, 100000, step=10000)
        kelly_fraction = st.slider("Kelly Fraction Multiplier", 0.1, 0.5, DEFAULTS['kelly_fraction'])
        max_position = st.slider("Max Position Size (%)", 0.05, 0.25, DEFAULTS['max_position'])
    
    cfg = {
        'n_states': n_states,
        'lookback_days': lookback_days,
        'min_probability': min_probability,
        'kelly_fraction': kelly_fraction,
        'max_position': max_position,
        'transition_threshold': 0.7
    }
    
    # Symbol input
    st.subheader("📊 Market Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Symbol", value="SPY", help="Enter stock ticker symbol (e.g., SPY, QQQ, AAPL)")
    with col2:
        analyze_button = st.button("🎯 Analyze Market Regime", type="primary")
    
    if analyze_button and symbol:
        with st.spinner(f"Analyzing {symbol} market regimes..."):
            # Get market data
            df = get_market_data(symbol)
            
            if df.empty:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Prepare features for Markov model
            feature_columns = ['returns', 'volatility', 'volume_ratio', 'price_momentum_5', 'price_momentum_20']
            features = df[feature_columns].values
            
            # Fit Markov model
            detector = MarkovRegimeDetector(n_states=cfg['n_states'])
            detector.fit(features)
            
            # Generate current signals
            signals = generate_markov_signals(df, detector, portfolio_value, cfg)
            
            # Store results in session state
            st.session_state.df = df
            st.session_state.detector = detector
            st.session_state.signals = signals
            st.session_state.symbol = symbol
    
    # Display results if available
    if 'df' in st.session_state and 'detector' in st.session_state:
        df = st.session_state.df
        detector = st.session_state.detector
        signals = st.session_state.signals
        symbol = st.session_state.symbol
        
        # Current market state
        st.subheader(f"📈 Current Market State for {symbol}")
        
        if signals:
            signal = signals[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Regime", 
                    STATE_NAMES[signal.current_state],
                    f"{signal.state_probability:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "Signal",
                    signal.signal_type,
                    f"Kelly: {signal.kelly_fraction:.2%}"
                )
            
            with col3:
                st.metric(
                    "Position Size",
                    f"{signal.position_size:,} shares",
                    f"${signal.position_size * df.iloc[-1]['Close']:,.0f}"
                )
            
            with col4:
                current_price = df.iloc[-1]['Close']
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{df.iloc[-1]['returns']:.2%} today"
                )
            
            # Signal reasoning
            st.info(f"**Reasoning:** {signal.reasoning}")
        
        # Charts
        st.subheader("📊 Market Regime Visualization")
        
        # Main regime chart
        regime_chart = create_regime_chart(df, detector)
        st.plotly_chart(regime_chart, use_container_width=True)
        
        # Transition matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔄 State Transition Matrix")
            transition_chart = create_transition_matrix_heatmap(detector)
            st.plotly_chart(transition_chart, use_container_width=True)
        
        with col2:
            st.subheader("📊 Regime Statistics")
            
            if detector.states is not None:
                # Calculate regime statistics
                state_counts = pd.Series(detector.states).value_counts().sort_index()
                total_days = len(detector.states)
                
                for state in range(detector.n_states):
                    if state in state_counts.index:
                        count = state_counts[state]
                        percentage = (count / total_days) * 100
                        
                        st.write(f"**{STATE_NAMES[state]}**")
                        st.write(f"- Days: {count} ({percentage:.1f}%)")
                        
                        if detector.transition_matrix is not None:
                            persistence = detector.transition_matrix[state, state]
                            st.write(f"- Persistence: {persistence:.1%}")
                        st.write("")
        
        # Performance metrics
        st.subheader("⚡ Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate regime-specific returns
            regime_returns = {}
            for state in range(detector.n_states):
                mask = detector.states == state
                if mask.sum() > 0:
                    returns = df['returns'][mask].mean()
                    regime_returns[state] = returns
            
            st.write("**Average Returns by Regime:**")
            for state, returns in regime_returns.items():
                st.write(f"{STATE_NAMES[state]}: {returns:.2%}")
        
        with col2:
            # Volatility by regime
            regime_vol = {}
            for state in range(detector.n_states):
                mask = detector.states == state
                if mask.sum() > 0:
                    vol = df['returns'][mask].std() * np.sqrt(252)  # Annualized
                    regime_vol[state] = vol
            
            st.write("**Volatility by Regime:**")
            for state, vol in regime_vol.items():
                st.write(f"{STATE_NAMES[state]}: {vol:.1%}")
        
        with col3:
            # Kelly sizing info
            if signals:
                signal = signals[0]
                risk_per_trade = portfolio_value * signal.kelly_fraction
                
                st.write("**Kelly Criterion:**")
                st.write(f"Fraction: {signal.kelly_fraction:.2%}")
                st.write(f"Risk per trade: ${risk_per_trade:,.0f}")
                st.write(f"Position limit: {cfg['max_position']:.1%}")
    
    # Information section
    with st.expander("ℹ️ About This Model"):
        st.write("""
        ### 🎯 Markov Chain Trading Model
        
        This application uses **Hidden Markov Models** to detect market regimes and the **Kelly Criterion** for position sizing, 
        inspired by Jim Simons' quantitative approach at Renaissance Technologies.
        
        **Key Features:**
        - **Regime Detection:** Identifies bull, bear, and sideways market states
        - **Transition Analysis:** Models probability of moving between states  
        - **Kelly Sizing:** Optimal position sizing based on historical performance
        - **Risk Management:** Built-in position limits and confidence thresholds
        
        **Model Inputs:**
        - Daily returns and volatility
        - Volume patterns and momentum
        - Technical indicators (RSI, moving averages)
        
        **⚠️ Important Disclaimers:**
        - This is for educational purposes only
        - Past performance does not guarantee future results
        - Always use proper risk management
        - Consider this one input among many for trading decisions
        
        **🧠 The Science:**
        Markov models assume that future market states depend only on the current state, not the entire history. 
        This "memoryless" property makes them powerful for regime detection while remaining computationally efficient.
        """)

if __name__ == "__main__":
    main()
