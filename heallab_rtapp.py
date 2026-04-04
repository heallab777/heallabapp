import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import streamlit.components.v1 as components
import time

# --- 1. 網頁配置 ---
st.set_page_config(
    page_title="Heal Lab | 聲香感官導航儀", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 初始化設定
if 'sensitivity' not in st.session_state: st.session_state.sensitivity = 75
if 'v_bias' not in st.session_state: st.session_state.v_bias = 0
if 'show_settings' not in st.session_state: st.session_state.show_settings = False

# --- 2. UI 注入 (Dark Mode 保持) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117 !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    
    /* 側邊欄/彈出面板深色化 */
    .settings-box {
        background: rgba(26, 30, 54, 0.98);
        border: 1px solid rgba(212, 180, 112, 0.5);
        border-radius: 20px; padding: 25px; margin-bottom: 25px;
    }
    
    /* 按鈕深灰色預設 */
    .stButton > button, .stDownloadButton > button {
        background-color: #262730 !important;
        color: #d4b470 !important;
        border: 1px solid rgba(212, 180, 112, 0.4) !important;
        border-radius: 12px !important;
    }
    
    /* 修正播放器在手機上的外觀 */
    audio { 
        filter: invert(100%) brightness(1.8) hue-rotate(180deg); 
        width: 100%; 
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. 浮動面板 ---
if st.button("⚙️ 參數校準 (Calibration)"):
    st.session_state.show_settings = not st.session_state.show_settings

if st.session_state.show_settings:
    with st.container():
        st.markdown('<div class="settings-box">', unsafe_allow_html=True)
        st.session_state.sensitivity = st.slider("能量增益", 10, 200, st.session_state.sensitivity)
        st.session_state.v_bias = st.slider("頻率偏好", -800, 800, st.session_state.v_bias)
        if st.button("💾 儲存設定"):
            st.session_state.show_settings = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. 運算函數 ---
def map_to_russell(energy, centroid, sens, bias):
    arousal = np.clip((energy * sens) - 0.6, -1, 1) 
    valence = np.clip(((2800 + bias) - centroid) / 2000, -1, 1)
    return valence, arousal

def draw_russell_chart(v, a):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='#ff4b4b', size=25, line=dict(color='white', width=3), symbol="diamond"),
        text=["當前狀態"], textposition="top center"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[-1.1, 1.1], zeroline=True),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True),
        shapes=[
            dict(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="rgba(150, 100, 255, 0.1)"),
            dict(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="rgba(100, 255, 100, 0.1)"),
            dict(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="rgba(255, 150, 50, 0.1)"),
            dict(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="rgba(210, 180, 140, 0.1)")
        ],
        margin=dict(l=10, r=10, t=10, b=10), height=400, showlegend=False
    )
    return fig

# --- 5. 主流程 ---
st.title("🌿 HEAL LAB | 導航儀")
tab1, tab2 = st.tabs(["🎤 即時感測", "📁 匯入分析"])

audio_data = None
with tab1:
    rec = mic_recorder(start_prompt="啟動感測 🎤", stop_prompt="停止並分析 ⏹️", key='fixed_recorder')
    if rec:
        audio_data = rec['bytes']

with tab2:
    up = st.file_uploader("匯入檔案", type=["mp3", "wav"])
    if up:
        audio_data = up.read()

if audio_data:
    try:
        # 使用 pydub 確保格式正確，並導出為標準 wav
        audio_io = io.BytesIO(audio_data)
        audio_seg = AudioSegment.from_file(audio_io)
        
        # 導出為標準 WAV 格式，這能確保 Metadata 完整，讓播放器顯示時間
        wav_buffer = io.BytesIO()
        audio_seg.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # 數據分析
        y, sr = librosa.load(wav_buffer)
        energy = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        v, a = map_to_russell(energy, centroid, st.session_state.sensitivity, st.session_state.v_bias)

        # --- 顯示播放器：回到原生方式，但強制指定格式 ---
        wav_buffer.seek(0)
        st.audio(wav_buffer, format="audio/wav")

        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(draw_russell_chart(v, a), use_container_width=True)
        with c2:
            st.subheader("🔍 診斷報告")
            if a > 0 and v < 0: st.warning("🌿 **建議：薰衣草**")
            elif a > 0 and v > 0: st.success("🍃 **建議：薄荷**")
            elif a < 0 and v > 0: st.info("🪵 **建議：檀香**")
            else: st.info("🍊 **建議：甜橙**")
            
            st.divider()
            st.download_button("💾 保存樣本", data=audio_data, file_name=f"heal_{int(time.time())}.wav")

    except Exception as e:
        st.error(f"分析異常：{e}")

st.caption("© 2026 Heal Lab | v5.5 Metadata Fixed")