import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import streamlit.components.v1 as components
import base64
import time

# --- 1. 網頁配置 ---
st.set_page_config(
    page_title="Heal Lab | 聲香感官導航儀", 
    layout="wide",
    initial_sidebar_state="collapsed" # 預設關閉原生側邊欄
)

# 初始化設定值
if 'sensitivity' not in st.session_state:
    st.session_state.sensitivity = 75
if 'v_bias' not in st.session_state:
    st.session_state.v_bias = 0
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# --- 2. 旗艦版 UI 注入 ---
st.markdown("""
    <style>
    /* 啟動動畫 */
    #splash-screen {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at center, #1a1e36 0%, #0d1117 100%);
        z-index: 9999; display: flex; justify-content: center; align-items: center;
        animation: fadeOut 3s forwards; pointer-events: none;
    }
    .splash-logo { color: #d4b470; font-size: 3.5em; font-weight: 900; letter-spacing: 10px; }
    @keyframes fadeOut { 0% { opacity: 1; } 100% { opacity: 0; visibility: hidden; } }

    /* 全域背景 */
    .stApp { background-color: #0d1117 !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }

    /* 浮動設定面板樣式 */
    .settings-box {
        background: rgba(26, 30, 54, 0.95);
        border: 1px solid rgba(212, 180, 112, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* 深灰色按鈕優化 */
    .stButton > button, .stDownloadButton > button {
        background-color: #262730 !important;
        color: #d4b470 !important;
        border: 1px solid rgba(212, 180, 112, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #d4b470 !important;
        color: #0d1117 !important;
    }

    /* 隱藏原生組件 */
    #MainMenu, footer, header, [data-testid="stSidebar"] {visibility: hidden;}
    </style>
    <div id="splash-screen"><div class="splash-logo">HEAL LAB</div></div>
    """, unsafe_allow_html=True)

# --- 3. 浮動控制邏輯 ---
col_set, col_empty = st.columns([1, 4])
with col_set:
    if st.button("⚙️ 開啟校準面板"):
        st.session_state.show_settings = not st.session_state.show_settings

if st.session_state.show_settings:
    with st.container():
        st.markdown('<div class="settings-box">', unsafe_allow_html=True)
        st.subheader("🎛️ 感官參數設定")
        st.session_state.sensitivity = st.slider("能量靈敏度 (Arousal)", 10, 200, st.session_state.sensitivity)
        st.session_state.v_bias = st.slider("頻率偏好 (Valence)", -800, 800, st.session_state.v_bias)
        
        if st.button("💾 儲存並關閉設定"):
            st.session_state.show_settings = False
            st.success("設定已存儲")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. 運算與繪圖 ---
def map_to_russell(energy, centroid, sens, bias):
    arousal = np.clip((energy * sens) - 0.6, -1, 1) 
    valence = np.clip(((2800 + bias) - centroid) / 2000, -1, 1)
    return valence, arousal

def draw_russell_chart(v, a):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='#ff4b4b', size=25, line=dict(color='white', width=3), symbol="diamond"),
        text=["當下狀態"], textposition="top center",
        textfont=dict(color="white", size=14)
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.8)', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.8)', gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=10, r=10, t=10, b=10), height=450, showlegend=False,
        shapes=[
            dict(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="rgba(150, 100, 255, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="rgba(100, 255, 100, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="rgba(255, 150, 50, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="rgba(210, 180, 140, 0.15)", line=dict(width=0))
        ],
        annotations=[
            dict(x=0.8, y=0.8, text="專注 (薄荷) 🍃", showarrow=False, font=dict(color="#ccffcc", size=15)),
            dict(x=-0.8, y=0.8, text="焦慮 (薰衣草) 🌿", showarrow=False, font=dict(color="#e6ccff", size=15)),
            dict(x=-0.8, y=-0.8, text="疲憊 (甜橙) 🍊", showarrow=False, font=dict(color="#ffe6cc", size=15)),
            dict(x=0.8, y=-0.8, text="放鬆 (檀香) 🪵", showarrow=False, font=dict(color="#f5deb3", size=15))
        ]
    )
    return fig

# --- 5. 主流程 ---
st.title("🌿 HEAL LAB | 感官導航儀")

tab1, tab2 = st.tabs(["🎤 即時採集", "📁 匯入音檔"])
audio_data = None

with tab1:
    rec = mic_recorder(start_prompt="啟動 AI 感測 🎤", stop_prompt="分析數據 ⏹️", key='recorder')
    if rec:
        audio_data = rec['bytes']

with tab2:
    up = st.file_uploader("匯入檔案", type=["mp3", "wav"])
    if up:
        audio_data = up.read()

if audio_data:
    try:
        with st.spinner('AI 正在感應聲波...'):
            audio_io = io.BytesIO(audio_data)
            audio_seg = AudioSegment.from_file(audio_io)
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            wav_io.seek(0)
            
            y, sr = librosa.load(wav_io)
            energy = np.mean(librosa.feature.rms(y=y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # 使用 session_state 儲存的設定值
            v, a = map_to_russell(energy, centroid, st.session_state.sensitivity, st.session_state.v_bias)

            wav_io.seek(0)
            b64 = base64.b64encode(wav_io.read()).decode()
            timestamp = int(time.time())
            
            audio_html = f"""
                <div style="background: rgba(212,180,112,0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(212,180,112,0.3); margin-bottom: 20px;">
                    <p style="color: #d4b470; font-size: 0.9em; margin-bottom: 8px;">🔊 最新樣本</p>
                    <audio controls key="{timestamp}" style="width: 100%;">
                        <source src="data:audio/wav;base64,{b64}#t={timestamp}" type="audio/wav">
                    </audio>
                </div>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

            col_chart, col_data = st.columns([1.5, 1])
            with col_chart:
                st.plotly_chart(draw_russell_chart(v, a), use_container_width=True)
            
            with col_data:
                st.subheader("🔍 診斷報告")
                if a > 0 and v < 0: st.warning("🌿 **建議：薰衣草**")
                elif a > 0 and v > 0: st.success("🍃 **建議：薄荷**")
                elif a < 0 and v > 0: st.info("🪵 **建議：檀香**")
                else: st.info("🍊 **建議：甜橙**")
                
                st.divider()
                st.download_button("💾 下載最新樣本", data=audio_data, file_name=f"heallab_{timestamp}.wav")

    except Exception as e:
        st.error(f"分析異常：{e}")

st.caption("© 2026 Heal Lab | Founder Edition v5.0 | Custom Controller")