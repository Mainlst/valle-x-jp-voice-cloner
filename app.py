import tempfile
import time
import os

# --- Monkey patch to fix Gradio schema parsing error ---
try:
    import gradio_client.utils as client_utils
    if not hasattr(client_utils, "_orig_json_schema_to_python_type"):
        client_utils._orig_json_schema_to_python_type = client_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        try:
            return client_utils._orig_json_schema_to_python_type(schema, defs)
        except Exception:
            return "Any"

    client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except ImportError:
    pass

# ZeroGPU / Spaces GPU decorator setup
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func:
                return func
            def wrapper(f):
                return f
            return wrapper

# ---- VALL‑E‑X utility modules ----
from utils.prompt_making import make_prompt
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
import gradio as gr
from scipy.io.wavfile import write as write_wav

# モデルをメモリにロードしておく（初回だけ時間が掛かります）
preload_models()

@spaces.GPU  # GPUを要求する関数を装飾
def clone_and_generate(uploaded_audio_path, transcript, script):
    """グラディオのコールバック関数。
    1. アップロードされた音声ファイルパスを取得
    2. make_prompt() で話者プロンプトを生成
    3. generate_audio() で台本を合成
    4. 合成音声を一時 WAV として書き出し、そのパスを返す
    """
    if uploaded_audio_path is None or not transcript or not script:
        return None, "⚠️ 音声・文字起こし・台本をすべて入力してください"

    # 入力ファイルパス
    audio_path = uploaded_audio_path

    # 一意なプロンプト名でクローンプロンプト生成
    prompt_name = f"prompt_{int(time.time())}"
    make_prompt(name=prompt_name, audio_prompt_path=audio_path, transcript=transcript)

    # 合成実行
    audio_array = generate_audio(script, prompt=prompt_name)

    # 出力WAV保存
    out_path = os.path.join(tempfile.gettempdir(), f"{prompt_name}_generated.wav")
    write_wav(out_path, SAMPLE_RATE, audio_array)

    return out_path, "✅ 生成完了しました！🎉"

# ---------------- Gradio UI 定義 ----------------
with gr.Blocks(title="VALL‑E‑X_JP-Voice-Cloner") as demo:
    gr.Markdown("""
    # 🎙️ VALL‑E‑X_JP-Voice-Cloner
    日本語対応の音声クローンアプリ（Hugging Face Spaces 向け）です。

    1. クローン元音声（1～3秒のWAV）をアップロード
    2. 文字起こしを入力
    3. 台本セリフを入力
    4. [🎙️ 音声生成]ボタンで合成結果が再生されます
    """)

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="① クローン元音声", type="filepath")
            transcript_in = gr.Textbox(label="② 文字起こし", lines=2, placeholder="例）これはテストの音声です。")
            script_in = gr.Textbox(label="③ 台本セリフ", lines=4, placeholder="例）今日はいい天気ですね、ご主人様。")
            generate_btn = gr.Button("🎙️ 音声生成", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="生成された音声", interactive=True)
            status = gr.Textbox(label="ステータス", interactive=False)

    generate_btn.click(fn=clone_and_generate,
                       inputs=[audio_in, transcript_in, script_in],
                       outputs=[output_audio, status])

if __name__ == "__main__":
    # Spaces 環境では share=True で公開リンクを生成
    demo.launch(share=True)
