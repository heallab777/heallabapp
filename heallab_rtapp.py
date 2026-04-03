import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import os
import streamlit.components.v1 as components  # 用於注入 JS

# --- 1. 網頁配置、PWA 封裝與高質感 UI ---
st.set_page_config(page_title="Heal Lab | 聲香感官系統", layout="wide", initial_sidebar_state="collapsed")

# 注入 CSS 與 PWA Meta

st.markdown("""
    <style>
    /* 1. 全域背景與動畫 (維持原本的高級感) */
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

    /* 2. 文字亮度大幅提升：改為亮白色與金色標題 */
    h1 {
        background: linear-gradient(to right, #d4b470, #ffffff) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
    }
    
    /* 讓所有說明文字、標籤變為亮白色，增加閱讀性 */
    h2, h3, p, span, label, .stMarkdown {
        color: #ffffff !important; 
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5); /* 加上輕微陰影增加對比 */
    }

    /* 3. 播放器 (Audio Player) 強制顯色 */
    /* 針對 Chrome/Edge/Safari 的原生播放器控制項進行濾鏡處理，讓它變亮 */
    audio {
        filter: invert(100%) hue-rotate(180deg) brightness(1.5);
        width: 100%;
        border-radius: 50px;
        margin-top: 10px;
    }

    /* 4. 玻璃擬態容器 */
    [data-testid="stVerticalBlock"] > div:has(div.stPlotlyChart), 
    .stTabs, .stAudio {
        background: rgba(255, 255, 255, 0.07) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* 5. 質感按鈕：維持發光感，但確保文字是深色清晰的 */
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        background: linear-gradient(90deg, #d4b470, #f0d9a0) !important;
        color: #1a1e36 !important; /* 按鈕文字用深藍色，對比度最高 */
        border-radius: 50px !important;
        box-shadow: 0 0 25px rgba(212, 180, 112, 0.5) !important;
    }

    /* 隱藏雜物 */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
# 顯示精煉品牌 Logo (金屬質感)
col_logo, _ = st.columns([1, 2])
with col_logo:
    # 假設您有一個品牌 Logo 圖片，這裡暫用 Emoji 與標題替代
    st.markdown("### 🌿 Heal Lab")

st.caption("創辦人：音響工程師 & 音樂創作者")
st.markdown("---")

# --- 2. 核心轉換函數 ---
def map_to_russell(energy, centroid):
    # 喚醒度 (Arousal): 能量越高點越往上
    arousal = np.clip((energy * 18) - 1, -1, 1) 
    # 愉悅度 (Valence): 穩定鋼琴往右，品質心往左
    valence = np.clip((2200 - centroid) / 1500, -1, 1)
    return valence, arousal

# --- 3. 繪製 Russell 座標圖 (更精緻的材質與光暈) ---
def draw_russell_chart(v, a):
    fig = go.Figure()
    
    # 材質優化：象限背景加入光暈補償
    fig.add_shape(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="#ccffcc", opacity=0.15, layer="below", line_width=0) # Green 專注
    fig.add_shape(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="#e6ccff", opacity=0.15, layer="below", line_width=0)   # Purple 焦慮
    fig.add_shape(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="#cce6ff", opacity=0.15, layer="below", line_width=0) # Blue 疲憊
    fig.add_shape(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="#ffe6cc", opacity=0.15, layer="below", line_width=0)  # Orange 放鬆

    # 加入目前情緒點
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='Red', size=18, line=dict(color='White', width=3)),
        text=["當下感官落點"], textposition="top center"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="愉悅度 (Valence)", range=[-1.1, 1.1], zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255, 255, 255, 0.2)'),
        yaxis=dict(title="喚醒度 (Arousal)", range=[-1.1, 1.1], zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255, 255, 255, 0.2)'),
        width=550, height=550, showlegend=False,
        annotations=[
            dict(x=0.7, y=0.9, text="專注 (薄荷) 🍃", showarrow=False, font=dict(color="#ccffcc", size=14)),
            dict(x=-0.7, y=0.9, text="焦慮 (薰衣草) 🌿", showarrow=False, font=dict(color="#e6ccff", size=14)),
            dict(x=-0.7, y=-0.9, text="疲憊 (甜橙) 🍊", showarrow=False, font=dict(color="#cce6ff", size=14)),
            dict(x=0.7, y=-0.9, text="放鬆 (檀香) 🪵", showarrow=False, font=dict(color="#ffe6cc", size=14))
        ]
    )
    return fig

# --- 4. JS 微指令：微動效與光暈補償 ---
def inject_aura_js(v, a):
    # 決定顏色 (1.5s ease 漸變)
    color = "transparent"
    if a > 0 and v < 0:
        color = "rgba(102, 0, 102, 0.3)"  # 焦慮 (深紫)
    elif a > 0 and v > 0:
        color = "rgba(0, 51, 0, 0.3)"    #專注 (亮綠)
    elif a < 0 and v > 0:
        color = "rgba(102, 51, 0, 0.3)"   # 放鬆 (暖橘)
    else:
        color = "rgba(0, 0, 102, 0.3)"   # 疲憊 (深藍)

    js_code = f"""
    <script>
    // 1.動態光暈補償
    parent.document.getElementById('background-aura').style.backgroundColor = '{color}';
    
    // 2.微動效 (Slide to location, 時間 1.5s)
    // 這裡只是設定一個目標值給父視窗的 JS 函數處理
    if (parent.updateEmotionPoint) {{
        parent.updateEmotionPoint({v}, {a});
    }}
    </script>
    """
    components.html(js_code, height=0)

# --- 5. 主介面設計 (全功能整合版) ---
tab1, tab2 = st.tabs(["🎤 即時錄音", "📁 上傳音檔"])

audio_source = None

with tab1:
    st.markdown("### 🎤 即時感官採集")
    st.write("請錄製 5-10 秒的環境音或音樂作品，AI 將為您媒合當下情緒。")
    # 這裡是 highlighted 的按鈕風格
    recorded_data = mic_recorder(start_prompt="錄音開始 ⏺️", stop_prompt="停止並分析 ⏹️", key='recorder')
    if recorded_data:
        audio_source = io.BytesIO(recorded_data['bytes'])

with tab2:
    st.markdown("### 📁 上傳音檔")
    st.write("上傳您之前錄製好的 MP3 或 WAV 檔案。")
    uploaded_file = st.file_uploader("選擇檔案", type=["mp3", "wav"])
    if uploaded_file:
        audio_source = uploaded_file

if audio_source:
    try:
        with st.spinner('AI 正在感應頻率中...'):
            audio_seg = AudioSegment.from_file(audio_source)
            wav_buffer = io.BytesIO()
            audio_seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            y, sr = librosa.load(wav_buffer)
            avg_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            energy = np.mean(librosa.feature.rms(y=y))
            v, a = map_to_russell(energy, avg_centroid)

            # --- 顯示區 ---
            # 播放器材质化
            st.markdown("### 🎵 播放分析音訊")
            st.audio(audio_source)

            col_left, col_right = st.columns([1.2, 1])
            
            with col_left:
                st.plotly_chart(draw_russell_chart(v, a))
                # 關鍵升級：注入 JS 微動效與光暈
                inject_aura_js(v, a)
            
            with col_right:
                st.markdown("### 🔍 AI 數據診斷")
                
                # 情緒對位指示器 (Aura)
                vol_bar = min(float(energy * 15), 1.0)
                st.write(f"當下能量強度：{'🔥' if vol_bar > 0.6 else '🌿'}")
                st.progress(vol_bar)
                
                # 科學數據與配方
                if a > 0 and v < 0:
                    st.warning("🌿 建議配方：【薰衣草】。偵測到高壓噪音，建議使用薰衣草舒緩神經。")
                elif a > 0 and v > 0:
                    st.success("🍃 建議配方：【薄荷】。進入專注心流，薄荷能協助您維持清透思緒。")
                elif a < 0 and v > 0:
                    st.info("🪵 建議配方：【檀香】。身心進入靜謐定錨區域，適合深度冥想。")
                else:
                    st.info("🍊 建議配方：【甜橙】。能量較低，適合使用柑橘調提升心情暖度。")

                st.divider()
                st.button("📦 取得此情緒專屬香氛貼")

    except Exception as e:
        st.error(f"分析發生錯誤：{e}")

else:
    st.info("💡 請選擇「錄音」或「上傳檔案」來啟動感官導航儀。")

# 頁尾一致性標誌
st.caption("© 2026 Heal Lab | 全球考察專用版本 v2.1")