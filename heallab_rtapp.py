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

# --- 2. 旗艦版 UI 注入 (包含啟動動畫、動態背景、亮白文字與播放器優化) ---
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
    @keyframes logoPulse {
        0% { opacity: 0; transform: scale(0.8); filter: blur(10px); }
        50% { opacity: 1; transform: scale(1); filter: blur(0px); }
        100% { opacity: 0; transform: scale(1.1); filter: blur(5px); }
    }
    .splash-logo {
        color: #d4b470;
        font-size: 3.5em;
        font-weight: 900;
        letter-spacing: 10px;
        text-shadow: 0 0 30px rgba(212, 180, 112, 0.6);
        animation: logoPulse 3s ease-in-out forwards;
    }
    @keyframes fadeOut {
        0% { opacity: 1; visibility: visible; }
        80% { opacity: 1; }
        100% { opacity: 0; visibility: hidden; }
    }

    /* 動態流體背景 */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #0d1117, #1a1e36, #2d1b4d, #0d1117) !important;
        background-size: 400% 400% !important;
        animation: gradientBG 15s ease infinite !important;
    }

    /* 文字與標題：強制亮白對比 */
    h1 {
        background: linear-gradient(to right, #d4b470, #ffffff) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
    }
    h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important; 
        text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
    }

    /* 播放器濾鏡優化 (讓按鈕變亮) */
    audio {
        filter: invert(100%) hue-rotate(180deg) brightness(1.5);
        width: 100%;
        border-radius: 50px;
    }

    /* 玻璃擬態容器 */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart), 
    .stTabs, .stAudio {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 25px !important;
        padding: 15px !important;
    }

    /* 金屬質感按鈕 */
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        background: linear-gradient(90deg, #d4b470, #f0d9a0) !important;
        color: #1a1e36 !important;
        border: none !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        box-shadow: 0 0 20px rgba(212, 180, 112, 0.5) !important;
    }

    /* 背景光暈容器 */
    #background-aura {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        transition: background-color 1.5s ease;
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

# --- 3. 側邊欄：靈敏度與校準控制 ---
with st.sidebar:
    st.markdown("### 🎛️ 感官校準面板")
    # 解決你提到的「低能量」問題，預設拉高到 60
    sensitivity = st.slider("能量靈敏度 (Arousal Gain)", 10, 200, 65)
    v_bias = st.slider("頻率偏好校準 (Valence Bias)", -800, 800, 0)
    st.divider()
    st.caption("Heal Lab Founder Edition v2.5")

# --- 4. 核心運算函數 ---
def map_to_russell(energy, centroid, sens, bias):
    # 喚醒度 (Arousal): 加上靈敏度調整
    arousal = np.clip((energy * sens) - 0.5, -1, 1) 
    # 愉悅度 (Valence): 加上頻率偏移校準
    valence = np.clip(((2200 + bias) - centroid) / 1500, -1, 1)
    return valence, arousal

def draw_russell_chart(v, a):
    fig = go.Figure()
    # 象限背景 (保持不變)
    fig.add_shape(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="#ccffcc", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="#e6ccff", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="#cce6ff", opacity=0.1, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="#ffe6cc", opacity=0.1, layer="below", line_width=0)

    # 修正後的目前情緒點：移除 shadow 屬性
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(
            color='#ff4b4b', 
            size=22, 
            line=dict(color='white', width=3)
            # 這裡移除了引發錯誤的 shadow 屬性
        ),
        text=["當下感官落點"], 
        textposition="top center",
        textfont=dict(color="white", size=14)
    ))

    # 以下圖表設定保持不變...
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', title="愉悅度 (Valence)"),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', title="喚醒度 (Arousal)"),
        showlegend=False,
        annotations=[
            dict(x=0.7, y=0.8, text="專注 (薄荷) 🍃", showarrow=False, font=dict(color="#ccffcc")),
            dict(x=-0.7, y=0.8, text="焦慮 (薰衣草) 🌿", showarrow=False, font=dict(color="#e6ccff")),
            dict(x=-0.7, y=-0.8, text="疲憊 (甜橙) 🍊", showarrow=False, font=dict(color="#cce6ff")),
            dict(x=0.7, y=-0.8, text="放鬆 (檀香) 🪵", showarrow=False, font=dict(color="#ffe6cc"))
        ]
    )
    return fig

# --- 5. JS 動態光暈補償 ---
def inject_aura_js(v, a):
    color = "transparent"
    if a > 0 and v < 0: color = "rgba(102, 0, 102, 0.4)"   # 焦慮-紫
    elif a > 0 and v > 0: color = "rgba(0, 80, 0, 0.4)"    # 專注-綠
    elif a < 0 and v > 0: color = "rgba(120, 60, 0, 0.4)"  # 放鬆-橘
    else: color = "rgba(0, 40, 100, 0.4)"                 # 疲憊-藍

    js_code = f"""
    <script>
    parent.document.getElementById('background-aura').style.backgroundColor = '{color}';
    </script>
    """
    components.html(js_code, height=0)

# --- 6. 主介面流程 ---
st.title("🌿 HEAL LAB | 感官導航系統")
st.write("透過聲音頻譜分析，為您媒合當下最適合的香氛配方。")

tab1, tab2 = st.tabs(["🎤 即時採集", "📁 匯入音檔"])
audio_source = None

with tab1:
    recorded_data = mic_recorder(start_prompt="開始感測 ⏺️", stop_prompt="停止分析 ⏹️", key='recorder')
    if recorded_data:
        audio_source = io.BytesIO(recorded_data['bytes'])

with tab2:
    uploaded_file = st.file_uploader("選擇您的音樂作品", type=["mp3", "wav"])
    if uploaded_file:
        audio_source = uploaded_file

if audio_source:
    try:
        with st.spinner('AI 正在感應聲波能量...'):
            audio_seg = AudioSegment.from_file(audio_source)
            wav_buffer = io.BytesIO()
            audio_seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            y, sr = librosa.load(wav_buffer)
            avg_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            energy = np.mean(librosa.feature.rms(y=y))
            
            # 使用滑桿變數進行運算
            v, a = map_to_russell(energy, avg_centroid, sensitivity, v_bias)

            st.audio(audio_source)

            col_chart, col_data = st.columns([1.5, 1])
            with col_chart:
                st.plotly_chart(draw_russell_chart(v, a), use_container_width=True)
                inject_aura_js(v, a)
            
            with col_data:
                st.markdown("### 🔍 診斷報告")
                st.write(f"當下偵測能量：**{energy:.4f}**")
                st.progress(min(float(energy * (sensitivity/10)), 1.0))
                
                if a > 0 and v < 0:
                    st.warning("🌿 **建議：薰衣草 (Lavender)**\n系統偵測到高壓頻率，建議透過乙酸芳樟酯平衡神經系統。")
                elif a > 0 and v > 0:
                    st.success("🍃 **建議：薄荷 (Peppermint)**\n您正處於高效心流狀態，薄荷能協助維持清透思緒。")
                elif a < 0 and v > 0:
                    st.info("🪵 **建議：檀香 (Sandalwood)**\n音頻趨於和諧平穩，適合使用木質調進行深度冥想。")
                else:
                    st.info("🍊 **建議：甜橙 (Sweet Orange)**\n感官能量偏低，建議使用柑橘類香氣提升心情暖度。")
                
                st.divider()
                if st.button("📦 取得此情緒專屬香氛貼"):
                    st.balloons()
                    st.success("已加入您的 Heal Lab 願望清單！")

    except Exception as e:
        st.error(f"分析失敗：{e}")
else:
    st.info("💡 錄音完成後，AI 將自動滑動至對應的情緒象限。")

st.caption("© 2026 Heal Lab | 專用版本")