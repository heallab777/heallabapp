import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import streamlit.components.v1 as components

# --- 1. 網頁配置與 PWA 封裝 ---
st.set_page_config(
    page_title="Heal Lab | 聲香感官導航儀", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. 旗艦版 UI 注入 (包含啟動動畫、動態背景、亮白文字) ---
st.markdown("""
    <style>
    /* 啟動動畫 (Splash Screen) */
    #splash-screen {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at center, #1a1e36 0%, #0d1117 100%);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        animation: fadeOut 3.5s forwards;
        pointer-events: none;
    }
    .splash-logo {
        color: #d4b470;
        font-size: 3.5em;
        font-weight: 900;
        letter-spacing: 10px;
        text-shadow: 0 0 30px rgba(212, 180, 112, 0.6);
        animation: logoPulse 3s ease-in-out forwards;
    }
    @keyframes logoPulse {
        0% { opacity: 0; transform: scale(0.8); filter: blur(10px); }
        50% { opacity: 1; transform: scale(1); filter: blur(0px); }
        100% { opacity: 0; transform: scale(1.1); filter: blur(5px); }
    }
    @keyframes fadeOut {
        0% { opacity: 1; visibility: visible; }
        80% { opacity: 1; }
        100% { opacity: 0; visibility: hidden; }
    }

    /* 動態流體背景 */
    .stApp {
        background: linear-gradient(-45deg, #0d1117, #1a1e36, #2d1b4d, #0d1117) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 文字強制亮白對比 */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important; 
        text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
    }
    h1 {
        background: linear-gradient(to right, #d4b470, #ffffff) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    /* 播放器優化：讓按鈕在深色背景清晰可見 */
    audio {
        filter: invert(100%) hue-rotate(180deg) brightness(1.8);
        width: 100%;
        margin-top: 10px;
    }

    /* 玻璃擬態容器 */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart), 
    .stTabs, .stAudio {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 20px !important;
    }

    /* 背景光暈效果 */
    #background-aura {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        transition: background-color 2s ease;
        pointer-events: none;
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    
    <div id="splash-screen">
        <div class="splash-logo">HEAL LAB</div>
        <div style="color:rgba(212,180,112,0.6); letter-spacing:4px;">SCENT & SOUND LABORATORY</div>
    </div>
    <div id="background-aura"></div>
    """, unsafe_allow_html=True)

# --- 3. 側邊欄：校準控制面板 ---
with st.sidebar:
    st.markdown("### 🎛️ 感官校準面板")
    # Arousal 增益：解決錄音能量太低的問題
    sensitivity = st.slider("能量靈敏度 (Arousal Gain)", 10, 200, 75)
    # Valence 偏移：解決環境高頻雜音太多的問題
    v_bias = st.slider("頻率偏好校準 (Valence Bias)", -1000, 1000, 0)
    st.divider()
    st.caption("Founder Edition v2.8 | Tokyo Ready")

# --- 4. 核心邏輯函數 ---
def map_to_russell(energy, centroid, sens, bias):
    # 提高能量計算的反應度
    arousal = np.clip((energy * sens) - 0.6, -1, 1) 
    # 將基準點設為 3000 以適應手機/筆電的高頻特性
    valence = np.clip(((3000 + bias) - centroid) / 2000, -1, 1)
    return valence, arousal

def draw_russell_chart(v, a):
    fig = go.Figure()
    # 象限裝飾背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=1.2, y1=1.2, fillcolor="#ccffcc", opacity=0.05, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-1.2, y0=0, x1=0, y1=1.2, fillcolor="#e6ccff", opacity=0.05, layer="below", line_width=0)
    
    # 目前情緒點 (移除引發報錯的 shadow)
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='#ff4b4b', size=25, line=dict(color='white', width=3)),
        text=["當下感官落點"], textposition="top center",
        textfont=dict(color="white", size=14)
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.2)'),
        showlegend=False,
        annotations=[
            dict(x=0.7, y=0.8, text="專注 (薄荷) 🍃", showarrow=False),
            dict(x=-0.7, y=0.8, text="焦慮 (薰衣草) 🌿", showarrow=False),
            dict(x=-0.7, y=-0.8, text="疲憊 (甜橙) 🍊", showarrow=False),
            dict(x=0.7, y=-0.8, text="放鬆 (檀香) 🪵", showarrow=False)
        ]
    )
    return fig

def inject_aura_js(v, a):
    color = "transparent"
    if a > 0 and v < 0: color = "rgba(138, 43, 226, 0.3)"   # 焦慮-紫
    elif a >