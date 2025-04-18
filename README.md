### 📄 `README.md`

```markdown
# 🎙️ VALL‑E‑X_JP-Voice-Cloner

日本語対応のゼロショット音声クローンアプリです。  
5秒ほどの音声サンプルと文字起こし＋台本テキストを入力するだけで、  
**話者の特徴を保持した新しいセリフ音声を合成**できます。

本プロジェクトは Microsoft の VALL‑E X を再現・日本語対応させた  
[Plachtaa/VALL-E-X](https://github.com/Plachtaa/VALL-E-X) に基づいています。

---

## 🐾 主な特徴

- 🇯🇵 **日本語音声対応**：日本語の音声・テキストでクローン可能
- 🧠 **ゼロショット合成**：話者情報は数秒の音声＋文字起こしだけ
- 🎭 **台本合成**：任意の台詞を任意の話者で喋らせられる
- 💻 **Gradio UI**：直感的なWebアプリ形式で提供
- 🐍 **Pythonベース**：ローカルでも動作・カスタマイズ可能

---

## 🚀 クイックスタート

### 1. 環境構築（conda推奨）

```bash
# Conda 環境の作成
conda env create -f environment.yaml
conda activate vallex-voice-cloner
```

または pip で：

```bash
pip install -r requirements.txt
```

### 2. アプリ起動

```bash
python app.py
```

ローカルブラウザで `http://localhost:7860` にアクセスできます。

---

## 🧪 使い方（Web UI）

1. 左側のパネルに以下を入力：
   - クローンしたい話者の音声（WAV, 1〜5秒）
   - その文字起こし
   - 話させたい台本テキスト
2. 「🎙️ 音声生成」ボタンをクリック
3. 右側に生成された音声が表示・再生可能になります

---

## 📁 ディレクトリ構成

```
.
├── app.py              # Gradio UI メインアプリ
├── checkpoints/        # 学習済みモデル (vallex-checkpoint.pt)
├── models/             # VALL‑E X モデル実装
├── utils/              # 前処理・合成・プロンプト処理ユーティリティ
├── requirements.txt    # pip 用依存
├── environment.yaml    # conda 用依存
├── README.md
└── ...
```

---

## 📜 ライセンス

- コード全体：MIT License  
- モデル・実装の一部は [Plachtaa/VALL-E-X](https://github.com/Plachtaa/VALL-E-X) を継承

> 本アプリは学術研究・実験用途を想定しています。商用利用は元ライセンスに従ってください。

---

## 🔗 参考文献・リソース

- 📄 [VALL-E X 論文](https://arxiv.org/abs/2303.03926)
- 🧠 [Plachtaa/VALL-E-X](https://github.com/Plachtaa/VALL-E-X)
- 🎧 [Facebook EnCodec](https://github.com/facebookresearch/encodec)
- 🗣️ [OpenAI Whisper](https://github.com/openai/whisper)
```