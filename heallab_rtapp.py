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
    initial_sidebar_state="collapsed"
)

# 初始化設定與音訊狀態
if 'sensitivity' not in st.session_state:
    st.session_state.sensitivity = 75
if 'v_bias' not in st.session_state:
    st.session_state.v_bias = 0
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None

# --- 2. 旗艦版 UI 注入 ---
st.markdown("""
    <style>
    #splash-screen {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at center, #1a1e36 0%, #0d1117 100%);
        z-index: 9999; display: flex; justify-content: center; align-items: center;
        animation: fadeOut 2s forwards; pointer-events: none;
    }
    .splash-logo { color: #d4b470; font-size: 3em; font-weight: 900; letter-spacing: 8px; }
    @keyframes fadeOut { 0% { opacity: 1; } 100% { opacity: 0; visibility: hidden; } }
    .stApp { background-color: #0d1117 !important; }
    h1, h2, h3, p, span, label { color: #ffffff !important; }
    
    /* 浮動面板 */
    .settings-box {
        background: rgba(26, 30, 54, 0.98);
        border: 1px solid rgba(212, 180, 112, 0.5);
        border-radius: 20px; padding: 25px; margin-bottom: 25px;
    }
    
    /* 按鈕與下載按鈕預設為深灰色 */
    .stButton > button, .stDownloadButton > button {
        background-color: #262730 !important;
        color: #d4b470 !important;
        border: 1px solid rgba(212, 180, 112, 0.4) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #d4b470 !important;
        color: #0d1117 !important;
    }

    /* 手機播放器修正 */
    audio { filter: invert(100%) brightness(1.8) hue-rotate(180deg); width: 100%; height: 50px; }
    
    .stTabs, [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart) {
        background: rgba(255, 255, 255, 0.03); border-radius: 20px; padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    <div id="splash-screen"><div class="splash-logo">HEAL LAB</div></div>
    """, unsafe_allow_html=True)

# --- 3. 浮動控制面板 ---
c_set, _ = st.columns([1, 3])
with c_set:
    if st.button("⚙️ 參數校準 (Calibration)"):
        st.session_state.show_settings = not st.session_state.show_settings

if st.session_state.show_settings:
    with st.container():
        st.markdown('<div class="settings-box">', unsafe_allow_html=True)
        st.subheader("🎛️ 聲學感測設定")
        st.session_state.sensitivity = st.slider("能量增益", 10, 200, st.session_state.sensitivity)
        st.session_state.v_bias = st.slider("頻率偏好", -800, 800, st.session_state.v_bias)
        if st.button("💾 儲存並關閉"):
            st.session_state.show_settings = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. 運算與圖表 ---
def map_to_russell(energy, centroid, sens, bias):
    arousal = np.clip((energy * sens) - 0.6, -1, 1) 
    valence = np.clip(((2800 + bias) - centroid) / 2000, -1, 1)
    return valence, arousal

def draw_russell_chart(v, a):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='#ff4b4b', size=25, line=dict(color='white', width=3), symbol="diamond"),
        text=["當前狀態"], textposition="top center",
        textfont=dict(color="white", size=14)
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.4)'),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.4)'),
        margin=dict(l=10, r=10, t=10, b=10), height=400, showlegend=False,
        shapes=[
            dict(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="rgba(150, 100, 255, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="rgba(100, 255, 100, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="rgba(255, 150, 50, 0.15)", line=dict(width=0)),
            dict(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="rgba(210, 180, 140, 0.15)", line=dict(width=0))
        ]
    )
    return fig

# --- 5. 主流程 ---
st.title("🌿 HEAL LAB | 導航儀")

tab1, tab2 = st.tabs(["🎤 即時感測", "📁 匯入分析"])

with tab1:
    # 修正：固定 Key 以確保錄音穩定性
    rec = mic_recorder(start_prompt="啟動感測 🎤", stop_prompt="停止並分析 ⏹️", key='fixed_recorder')
    if rec:
        st.session_state.last_audio = rec['bytes']

with tab2:
    up = st.file_uploader("匯入檔案", type=["mp3", "wav"])
    if up:
        st.session_state.last_audio = up.read()

# --- 6. 核心顯示區 ---
if st.session_state.last_audio:
    try:
        audio_bytes = st.session_state.last_audio
        with st.spinner('AI 正在計算聲波...'):
            audio_io = io.BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_file(audio_io)
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            wav_io.seek(0)
            
            y, sr = librosa.load(wav_io)
            energy = np.mean(librosa.feature.rms(y=y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            v, a = map_to_russell(energy, centroid, st.session_state.sensitivity, st.session_state.v_bias)

            # --- 播放器修復：Base64 數據封裝 ---
            wav_io.seek(0)
            b64 = base64.b64encode(wav_io.read()).decode()
            ts = int(time.time())
            
            audio_html = f"""
                <div style="background: rgba(212,180,112,0.1); padding: 15px; border-radius: 15px; border: 1px solid rgba(212,180,112,0.3); margin: 15px 0;">
                    <p style="color: #d4b470; font-size: 0.8em; margin-bottom: 8px;">🔊 最新樣本 (若無聲請按播放兩次)</p>
                    <audio controls key="{ts}" style="width: 100%;">
                        <source src="data:audio/wav;base64,{b64}#t={ts}" type="audio/wav">
                    </audio>
                </div>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.plotly_chart(draw_russell_chart(v, a), use_container_width=True)
            with c2:
                st.subheader("🔍 媒合結果")
                if a > 0 and v < 0: st.warning("🌿 **建議：薰衣草**")
                elif a > 0 and v > 0: st.success("🍃 **建議：薄荷**")
                elif a < 0 and v > 0: st.info("🪵 **建議：檀香**")
                else: st.info("🍊 **建議：甜橙**")
                
                st.divider()
                st.download_button("💾 下載檔案", data=audio_bytes, file_name=f"heal_{ts}.wav")

    except Exception as e:
        st.error(f"分析異常：{e}")
else:
    st.info("💡 提示：點擊錄音後發聲。手機端請確認非靜音模式。")

st.caption("© 2026 Heal Lab | v5.2 Stable Edition")