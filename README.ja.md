[English](./README.md) | [한국어](./README.ko.md) | [日本語](./README.ja.md)

---

# 🧠 JLPT問題生成学習ヘルパー - AIサービス

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](#-tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](#-tech-stack)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-purple.svg)](#-tech-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

「生成AIによるJLPT問題生成学習ヘルパー」の中核を担うAIサービスです。FastAPIをベースに構築され、LangChainを活用して様々なLLM(大規模言語モデル)と対話します。RAG(検索拡張生成)アーキテクチャを通じて、より正確で文脈に合ったJLPT問題の生成とチャットボットの応答機能を提供します。

## ✨ 主な機能

- **🤖 動的な問題生成**: LLMを活用し、JLPT試験タイプ(語彙、文法、読解)に合わせた問題をリアルタイムで生成。
- **💬 RAGベースのチャットボット**: FAISSベクトルストアに保存された知識ベースを活用し、ユーザーの質問に対して正確で根拠のある回答を提供。
- **🔄 マルチLLMサポート**: OpenAI, Google Gemini, Groqなど、必要に応じて様々な言語モデルを柔軟に切り替えて使用可能。
- **⚡️ 高性能な非同期API**: FastAPIにより、高いスループットと高速な応答速度を保証。

## 🏛️ アーキテクチャ: RAG (検索拡張生成)

このサービスは、RAG(Retrieval-Augmented Generation)アーキテクチャを採用してLLMの限界を補完します。

1.  **入力 (Input)**: ユーザーが問題生成リクエストまたは質問を入力します。
2.  **検索 (Retrieve)**: 入力内容と最も関連性の高い文書を`FAISS`ベクトルストアから検索します。
3.  **拡張 (Augment)**: 検索された文書(コンテキスト)と元の質問をプロンプトにまとめて、LLMに渡す準備をします。
4.  **生成 (Generate)**: 拡張されたプロンプトを`LangChain`を介してLLM(例: GPT-4, Gemini)に渡し、文脈に合った正確な回答または問題を生成します。

この方式により、ハルシネーション(幻覚)現象を減らし、特定ドメイン(JLPT)に対する専門性の高い結果を得ることができます。

## 🛠️ 技術スタック

| 区分 | 技術 / ライブラリ | 説明 |
| :--- | :--- | :--- |
| **言語** | Python | 3.12 |
| **ウェブフレームワーク** | FastAPI, Uvicorn | 非同期APIサーバー |
| **AIフレームワーク** | LangChain | LLMアプリケーション開発 |
| **LLM連携** | OpenAI, Google GenAI, Groq | |
| **ベクトル検索** | FAISS (faiss-cpu) | RAGのための埋め込みベクトル検索 |
| **環境変数管理**| python-dotenv | |
| **データ処理** | Pydantic, unstructured | |

## 📂 プロジェクト構造

```
app/
├── main.py                   # FastAPIアプリケーションのエントリーポイントとルーター設定
├── chatbot/                  # 一般的なチャットボット関連ロジック
├── problem_generator/        # JLPT問題生成ロジック
└── user_question_chatbot/    # ユーザーの質問に答えるRAGチャットボットロジック
```

## 🚀 始め方

### 1. 事前要件

- Python 3.12 以上
- pip

### 2. インストール

プロジェクトのルートディレクトリで以下のコマンドを実行し、依存関係をインストールします。
```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

プロジェクトのルートに`.env`ファイルを作成し、使用するLLMのAPIキーを入力します。

```
# .env

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key_here"

# Google GenAI API Key
GOOGLE_API_KEY="your_google_api_key_here"

# Groq API Key
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. 開発サーバーの実行

以下のコマンドを実行すると、Uvicorn開発サーバーが起動します。`--reload`オプションにより、コード変更時にサーバーが自動的に再起動します。
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 APIドキュメントとエンドポイント

FastAPIはOpenAPI 3.0仕様に準拠したAPIドキュメントを自動的に生成します。開発サーバー実行後、ウェブブラウザで**`http://localhost:8000/docs`**にアクセスすると、Swagger UIを介してすべてのAPIを確認し、直接テストすることができます。

- `POST /generate-problem`: 新しいJLPT問題の生成をリクエストします。
- `POST /chat`: RAGベースのチャットボットに質問します。

## 🐳 Dockerで実行

1.  **Dockerイメージをビルド**
    ```bash
    docker build -t jlpt-ai-service:latest .
    ```

2.  **Dockerコンテナを実行**
    `.env`ファイルのAPIキーを環境変数として注入してコンテナを実行します。
    ```bash
    docker run -p 8000:8000 \
      -e OPENAI_API_KEY="your_openai_api_key" \
      -e GOOGLE_API_KEY="your_google_api_key" \
      -e GROQ_API_KEY="your_groq_api_key" \
      jlpt-ai-service:latest
    ```

## 🤝 貢献

貢献はいつでも大歓迎です！Issueを作成するか、Pull Requestを送ってください。

## 📄 ライセンス

このプロジェクトはMITライセンスに従います。詳細については`LICENSE`ファイルを参照してください。
