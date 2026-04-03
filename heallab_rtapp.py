import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import os

# --- 1. 網頁配置與環境設定 ---
st.set_page_config(page_title="Heal Lab | 聲香感官導航", layout="wide")

# 強制指定 ffmpeg 路徑 (若 exe 在程式目錄下)
if os.path.exists("ffmpeg.exe"):
    AudioSegment.converter = os.getcwd() + "\\ffmpeg.exe"

# --- 2. 核心轉換函數 ---
def map_to_russell(energy, centroid):
    # 喚醒度 (Arousal): 能量越高點越往上
    arousal = np.clip((energy * 18) - 1, -1, 1) 
    # 愉悅度 (Valence): 低頻穩定往右(放鬆)，高品質心往左(焦慮)
    valence = np.clip((2200 - centroid) / 1500, -1, 1)
    return valence, arousal

# --- 3. 繪製 Russell 座標圖 ---
def draw_russell_chart(v, a):
    fig = go.Figure()
    # 象限背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=1.1, y1=1.1, fillcolor="LightYellow", opacity=0.3, layer="below", line_width=0) 
    fig.add_shape(type="rect", x0=-1.1, y0=0, x1=0, y1=1.1, fillcolor="Thistle", opacity=0.3, layer="below", line_width=0)   
    fig.add_shape(type="rect", x0=-1.1, y0=-1.1, x1=0, y1=0, fillcolor="LightSteelBlue", opacity=0.3, layer="below", line_width=0) 
    fig.add_shape(type="rect", x0=0, y0=-1.1, x1=1.1, y1=0, fillcolor="NavajoWhite", opacity=0.3, layer="below", line_width=0)  

    # 標示點
    fig.add_trace(go.Scatter(
        x=[v], y=[a], mode='markers+text',
        marker=dict(color='Red', size=18, line=dict(color='White', width=3)),
        text=["當下感官落點"], textposition="top center"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(title="愉悅度 (Valence)", range=[-1.1, 1.1], zeroline=True, zerolinewidth=2),
        yaxis=dict(title="喚醒度 (Arousal)", range=[-1.1, 1.1], zeroline=True, zerolinewidth=2),
        width=550, height=550, showlegend=False,
        annotations=[
            dict(x=0.7, y=0.9, text="專注 (薄荷) 🍃", showarrow=False),
            dict(x=-0.7, y=0.9, text="焦慮 (薰衣草) 🌿", showarrow=False),
            dict(x=-0.7, y=-0.9, text="疲憊 (甜橙) 🍊", showarrow=False),
            dict(x=0.7, y=-0.9, text="放鬆 (檀香) 🪵", showarrow=False)
        ]
    )
    return fig

# --- 4. 主介面設計 ---
st.title("🧭 Heal Lab：全功能聲香感官系統")
st.markdown("---")

tab1, tab2 = st.tabs(["🎤 即時錄音分析", "📁 上傳音檔分析"])

audio_source = None

# --- 分頁一：錄音 ---
with tab1:
    st.write("請錄製 5-10 秒的環境音或音樂作品：")
    recorded_data = mic_recorder(start_prompt="開始錄音 ⏺️", stop_prompt="停止錄音並分析 ⏹️", key='recorder')
    if recorded_data:
        audio_source = io.BytesIO(recorded_data['bytes'])

# --- 分頁二：上傳 ---
with tab2:
    st.write("上傳您創作的 MP3 或 WAV 檔案：")
    uploaded_file = st.file_uploader("選擇檔案", type=["mp3", "wav"])
    if uploaded_file:
        audio_source = uploaded_file

# --- 5. 統一處理邏輯 ---
if audio_source:
    try:
        with st.spinner('AI 正在感應頻率中...'):
            # 使用 pydub 進行格式標準化
            audio_seg = AudioSegment.from_file(audio_source)
            wav_buffer = io.BytesIO()
            audio_seg.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            # Librosa 讀取
            y, sr = librosa.load(wav_buffer)
            
            # 特徵計算
            avg_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            energy = np.mean(librosa.feature.rms(y=y))
            v, a = map_to_russell(energy, avg_centroid)

            # --- 顯示區 ---
            st.success("分析完成！")
            
            # 播放器
            st.subheader("🎵 播放分析音訊")
            st.audio(audio_source)

            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.plotly_chart(draw_russell_chart(v, a))
            
            with col_right:
                st.subheader("🛠️ 診斷與配方建議")
                vol_bar = min(float(energy * 15), 1.0)
                st.write(f"當下能量強度：{'🔥' if vol_bar > 0.6 else '🌿'}")
                st.progress(vol_bar)
                
                # 最終邏輯判斷
                if a > 0 and v < 0:
                    st.warning("### 🌿 【薰衣草配方】")
                    st.write("**狀態：焦慮。** 偵測到高壓噪音，建議立即使用薰衣草舒緩神經。")
                elif a > 0 and v > 0:
                    st.success("### 🍃 【薄荷配方】")
                    st.write("**狀態：專注。** 偵測到高效創作頻率，薄荷能協助您維持清透思緒。")
                elif a < 0 and v > 0:
                    st.info("### 🪵 【檀香配方】")
                    st.write("**狀態：放鬆。** 進入定錨區域，適合搭配木質香調深層冥想。")
                else:
                    st.info("### 🍊 【甜橙配方】")
                    st.write("**狀態：低迷。** 能量點較低，建議使用柑橘調提升心情愉悅感。")

                st.divider()
                st.button("📦 取得此情緒專屬香氛貼")

    except Exception as e:
        st.error(f"分析發生錯誤：{e}")
        st.info("提示：若上傳大檔案請稍候；若錄音失敗請確認麥克風權限。")

else:
    st.info("💡 請選擇「錄音」或「上傳檔案」來啟動感官導航儀。")

st.caption("© 2026 Heal Lab | 創辦人：音響工程師 & 音樂創作者")