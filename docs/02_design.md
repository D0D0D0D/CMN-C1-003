# CMN-C1-003 ChatAgent — 設計仕様書

**Template ID:** CMN-C1-003
**Agent Name:** ChatAgent
**カテゴリ:** Cat 1（単一の技術能力＝多ターン会話を汎用提供。「何ができるか」で定義。Industry 軸 CMN とは独立）
**継承モデル:** **L1 直接継承**（`AgentBaseGraph` from `framework/`）— 2026-05-18 のレイヤ方針改訂に準拠
**ステータス:** 設計フェーズ
**作成日:** 2026-06-01
**作成者:** 開発者AAA
**参考テンプレート:** AlertNotificationAgent（同じ固定パイプライン + L1 直接継承の構造アナログ）

---

> **📌 Cat 定義改訂への適合（2026-06-01 改訂）**
> Cat は「ビジネス観点の技術能力」で区分し、Industry 軸（CMN/RET 等）とは **独立**です
> （従来の「ドメイン非依存（CMN）＝ Cat 1」という運用は誤り）。
> - **Cat 1** = 単一の技術能力を汎用提供（例: VectorRAGAgent / WebSearchAgent / CodeGenerationAgent）。「何ができるか」で定義
> - **Cat 2** = 特定ユースケース文脈で複数ステップを組合せ仕事を完遂（例: SRSGapDetectionAgent）。「どの仕事を完遂するか」で定義
> - **Cat 3** = think-act-observe の自律ループ
>
> `ChatAgent` は **「多ターン会話」という単一の技術能力を汎用提供**するため、改訂後の定義でも **Cat 1 で確定**します。
> 内部に 4 ノードのパイプラインを持ちますが、これは単一能力（会話）の内部機構であり、
> 「特定ユースケースで仕事を完遂する」Cat 2 とは異なります（VectorRAGAgent が内部に retrieve→generate を持つのと同じ）。
> 注: テンプレート ID のリナンバリングは品質管理プロセス側で行い、開発側では変更しません。

---

## 1. 概要

`ChatAgent` は、**「多ターン会話」という単一の技術能力を汎用提供**する Cat 1 エージェントです
（Industry 軸では CMN ＝業種非依存だが、これは Cat 区分とは独立した属性）。
ユーザー発話（`user_input`）とセッション識別子（`session_id`）を受け取り、
**直近 N ターンの履歴ウィンドウ** と `system_prompt` を文脈として **単一 LLM 推論** を行い、
S-3 出力ゲートを通過した応答を返します。

**設計ゴール:**

- **汎用の会話エンジン（単一技術能力）** — 全業界の会話型派生テンプレートが本能力を再利用できる（system_prompt 差分だけの設定インスタンスは Cat 1 のまま。会話に複数ステップを足して特定の仕事を完遂するものが Cat 2）
- **設定のみで運用可能** — シチズン開発者が `system_prompt` + `history_window_turns` の指定だけでデプロイ可能
- **マルチセッション安全** — 顧客 A↔B の履歴混線を構造的に禁止
- **検索なし・ツールなし** — 純粋会話に責務を限定（RAG は VectorRAGAgent、DocGen は DocGenerationAgent が担当）

**スコープ（含む）:** 入力検証 → セッション初期化・履歴ロード → 履歴ウィンドウ管理 →
単一 LLM 推論 → 応答整形 → S-3 出力ゲート → 履歴永続化 → 監査トレース

**スコープ（含まない）:** ベクトル検索 / KB 参照（VectorRAGAgent RAG）、構造化文書生成（DocGenerationAgent DocGen）、
外部ツール呼び出し・自律実行（ReActAgent ReAct）

**Single-responsibility:** 本テンプレは「**多ターン会話の文脈管理 + 単一 LLM 推論**」専門。
検索・文書生成・ツール実行は混在させない（Dual-mode アンチパターンを回避）。

---

## 2. レイヤーアーキテクチャ

> ⚠️ **レイヤ方針注記（2026-05-18）:** テンプレートはすべて
> **L1 `framework/` を直接継承** する方針に変更されました。
> 提案時に当初想定していた **Level 2: ChatAgent base** は、継承ターゲットとしては廃止され、
> CMN-C1-003 自身が「会話型パターンのリファレンス実装（概念パターン）」として後発の会話型派生テンプレート（Cat 1 設定インスタンス / Cat 2 のいずれも）に参照されます。
> 本テンプレートは新方針に従い **L1 直接継承** で実装します（旧 `from agents.base.chat_agent import ...` の
> 継承パターンは使用しません）。

```
L0: agenticstar-platform SDK          ← 直接 import 禁止（import isolation ルール）
L1: framework/                        ← 直接継承（AgentBaseGraph + BaseNode）
CMN-C1-003                            ← 本テンプレート実装
cli.py                                ← AGENTIC STAR Marketplace entry adapter（scope 再構築 + Azure LLM 注入）
```

**継承の実装（SummarizationAgent リリース版に準拠）:**

Agent は **Graph を継承**する（`self.graph` 合成は不可 — 独自クラスは 5 層セキュリティを迂回する）。
production 実行は `AgentBaseGraph` から継承した **`invoke(state, config)`** を使う（`run()` は本番に存在せず
使用禁止 — 呼ぶと AttributeError になり S-1 trust ゲートも通らない。独自 `run()` / `invoke()` も定義しない）。`invoke()` は内部の
`_invoke_impl` で各ノードを `__call__` 経由に走らせ、各ノードの `required_trust_level` を S-1 自動強制する。

```python
# src/graph/graph.py — グラフ定義（ノード登録 + エッジ配線）
from framework.graph.agent_base_graph import AgentBaseGraph
from src.nodes.session_init import SessionInitNode
from src.nodes.history_manager import HistoryManagerNode
from src.nodes.llm_inference import LLMInferenceNode
from src.nodes.response import ResponseNode

class ChatGraph(AgentBaseGraph):
    """固定 4 ノード線形パイプライン。"""
    def register_nodes(self) -> None:
        self.add_node("session_init",    SessionInitNode())
        self.add_node("history_manager", HistoryManagerNode())
        self.add_node("llm_inference",   LLMInferenceNode())
        self.add_node("response",        ResponseNode())
    def add_edges(self) -> None:
        self.add_edge("session_init",    "history_manager")
        self.add_edge("history_manager", "llm_inference")
        self.add_edge("llm_inference",   "response")
        self.set_entry_point("session_init")
        self.set_finish_point("response")

# src/agent.py — Agent は Graph を継承し、S-1〜S-5 のフックを実装
from framework.schemas.invocation_context import InvocationContext, TrustLevel
from src.graph.graph import ChatGraph

REQUIRED_TRUST_LEVEL = TrustLevel.VERIFIED_EXTERNAL   # =1。agent 宣言値（config/agent.yaml と一致）

class ChatAgent(ChatGraph):
    """Cat 1 多ターン会話エージェント。Graph を継承し invoke() を継承する（run() は使わない）。"""
    TEMPLATE_ID = "CMN-C1-003"
    # _security_gate_input / _security_gate_output / _emit_trace_event を実装（§8）
    # S-1（Trust）は各ノードの required_trust_level を BaseNode.__call__ が自動強制（ADR-006）。
    # production エントリは invoke()。agent 側 __pre_invoke__ / run() は持たない（本番に run() は存在しない）
```

> **Trust レベルの根拠:** 本番 `TrustLevel` enum（IntEnum）は **`ANONYMOUS`（=0） / `VERIFIED_EXTERNAL`（=1） / `INTERNAL`（=2）** の 3 値（2026-06-01 発効）。`AUTHENTICATED` / `VERIFIED_INTERNAL` は存在しない旧エイリアスで使用禁止（import 時 AttributeError）。ChatAgent は認証済み外部エンドユーザー（顧客）の会話を
> 想定するため最小要件は **`VERIFIED_EXTERNAL`（=1）**。ANONYMOUS（=0）は拒否（default-deny）。
> 社内専用の派生テンプレート（社内ヘルプデスク等）は `INTERNAL`（=2）に昇格可。

---

## 3. ノードフロー

```
[入力: state["user_input"]（str）, state["session_id"]（str）]
      ↓
[SessionInitNode]     — (S-1 は __call__ が自動強制) S-2 入力ゲート + 履歴ロード + system_prompt 解決
      ↓
[HistoryManagerNode]  — 直近 N ターンのウィンドウ管理 + LLM 入力 messages 構築
      ↓
[LLMInferenceNode]    — 単一 LLM 呼び出し（検索なし・ツールなし）
      ↓
[ResponseNode]        — 応答整形 + S-3 出力ゲート + 履歴永続化 + S-4 監査
      ↓
→ state["formatted_output"]（呼び出し元へ返却）
```

### 各ノードの責務

| ノード | 責務 |
|---|---|
| `SessionInitNode` | `user_input` 検証（空文字拒否）+ S-2 入力ゲート（credential/injection scan）。`session_id` で `ConversationRepository.load()` から履歴ロード。`config.system_prompt_template` を解決して `system_prompt` に格納 |
| `HistoryManagerNode` | ロード済み履歴に今回の user turn を追加し、`config.history_window_turns`（直近 N ターン）でトリミング。LLM 入力用の `messages`（system + windowed history + user）を構築 |
| `LLMInferenceNode` | `system_prompt` + `messages` で単一 LLM 呼び出し。**KB 検索・ツール呼び出しは一切行わない**。`llm_response` を生成。`config.model` / `temperature` / `max_tokens` 適用 |
| `ResponseNode` | `llm_response` を整形。**S-3 出力ゲート（`_security_gate_output`）を適用**し credential/PII をマスク／ブロック。通過後、assistant 応答を履歴に追記し `ConversationRepository.save()`。S-4 監査トレースを emit |

### この 4 ノード分割の根拠

| 観点 | 理由 |
|---|---|
| **SessionInit と HistoryManager の分離** | セッション境界（どの履歴をロードするか）と、ウィンドウ整形（どこまで LLM に渡すか）を分離。マルチセッション安全の検証点を SessionInit に閉じ込められる |
| **LLMInference 単独** | 純粋な LLM 呼び出しに責務を限定。派生テンプレートが model/prompt を override する面を明確化 |
| **Response に S-3 + 永続化を集約** | 出力ゲートと履歴 write を 1 ノードに集約し、「ゲート未通過の応答は永続化も返却もしない」を構造的に保証 |

固定パイプライン（条件分岐なし）。エラー伝播は各ノードの `error_code` early-exit パターン（AlertNotificationAgent と同方式）。

---

## 4. State スキーマ（flat TypedDict、msgpack serializable）

```python
from typing import Optional
from framework.schemas.agent_state import AgentState

class ChatState(AgentState):
    # ── 入力 ──
    user_input:        Optional[str]   # 今回のユーザー発話
    session_id:        Optional[str]   # セッション識別子（マルチセッション分離キー）

    # ── SessionInitNode 出力 ──
    validated_input:   Optional[str]   # 検証・正規化済み入力
    system_prompt:     Optional[str]   # config.system_prompt_template を解決した system prompt
    conversation_history: Optional[str]  # JSON 配列 [{role, content}] — ロード済み履歴

    # ── HistoryManagerNode 出力 ──
    windowed_messages: Optional[str]   # JSON 配列 — LLM 入力用 messages（直近 N ターン + 今回）
    history_turn_count: Optional[int]  # ウィンドウ後のターン数

    # ── LLMInferenceNode 出力 ──
    llm_response:      Optional[str]   # LLM 生応答

    # ── ResponseNode 出力 ──
    formatted_output:  Optional[str]   # S-3 通過後の最終応答（呼び出し元へ返却）
    history_persisted: Optional[bool]  # 履歴永続化の成否

    # ── エラー伝播 ──
    error_code:        Optional[str]
    error_message:     Optional[str]
```

**State 安全則:**
- `conversation_history` / `windowed_messages` は **JSON 文字列**で保持（Pydantic/dataclass 禁止、msgpack 互換）
- JWT / API キー / credential は State に **書かない**。LLM 認証は `InvocationContext` 経由
- `InvocationContext` は State に格納せず `config["configurable"]["invocation_context"]` から参照
- 会話履歴に PII が含まれうるため、checkpoint 安全（PB-5）でセッション越え漏洩がないことを検証

---

## 5. Config 仕様（`config/agent.yaml`）

```yaml
template_id: CMN-C1-003
agent_name: ChatAgent
version: 1.0.0

# ─── 会話設定 ─────────────────────────────────
system_prompt_template: |
  あなたは親切で簡潔な会話アシスタントです。
  会話履歴の文脈を踏まえて、明確かつ簡潔に応答してください。
history_window_turns: 10          # 直近 N ターンを上限とする（古いターンから drop）
history_token_budget: 3000        # かつトークン予算で切り詰め（モデル上限に正確に収める。優先される実切り詰め基準）

# ─── LLM 設定 ─────────────────────────────────
model: claude-sonnet-4-6
temperature: 0.7
max_tokens: 1024
timeout_s: 30
max_retry: 3

# ─── セキュリティ ─────────────────────────────
required_trust_level: VERIFIED_EXTERNAL        # =1。ANONYMOUS は拒否（default-deny）
s3_gate_enabled: true                          # ❗ 絶対 false にしない
mask_patterns:                                 # ドメイン PII の拡張リスト（最小限。派生で追加）
  - "account_number"
  - "mynumber"
  - "appi_sensitive"
# NOTE: API キー / トークン / パスワード / 秘密鍵 等のシークレットは framework の S-2/S-3
#       credential scan が広くカバー（sk- / Bearer / api_key=secret / BEGIN PRIVATE KEY 等）。
#       クレジットカード番号など追加 PII が必要な派生は mask_patterns を拡張する。
#       標準 ChatAgent では PII を網羅列挙せず、最小セット + framework scan に委ねる。

# ─── 監査ログ（S-4） ─────────────────────────
audit_log_fields:
  - "template_id"
  - "session_id"            # セッション ID は記録、発話本文は記録しない
  - "history_turn_count"
  - "model"
# NOTE: user_input / llm_response の本文は NEVER ログしない（ターン数・トークン数のみ）
```

---

## 5-bis. LLM 注入シーム（`config["llm"]`）

LLM は **`config["llm"]` 経由でのみ**受け取る。モデル ID も具体クライアントも `src/` に
書かない（同系テンプレート共通）。AGENTIC STAR (Marketplace) では汎用 entry adapter が供給し、標準
`src/api/server.py` / CI では未注入。

### 🔴 旧実装の欠陥（本節が是正するもの）

外側 Graph が `ChatGraph(config, repository, llm_client)` という**独自コンストラクタ引数**で
LLM を受け取っていた。framework の実契約が想定しないパターンで、書かれているとおりの
結果になっていた:

- `src/api/server.py` は framework の実契約（`BaseGraph.__init__(self, config=None)`）に
  従い `ChatAgent(config=_config)` としか呼ばない → `llm_client` は**永久に未設定**
- 既定ビルドは `LLMClient.generate()` の `NotImplementedError` を踏み、**毎回
  `LLM_CALL_FAILED`** を返していた（`generation_mode: "llm"` の宣言は実体を伴っていなかった）
- テストが緑だったのは `tests/conftest.py` が `agent._llm_client` を**private 属性で直接
  書き換えて**いたため。本番経路は一度も通っていなかった

### プロトコル差の吸収

注入されるのは `BaseLLM`（`complete(messages) -> dict`）で、本 AG のノードが呼ぶのは
`generate(prompt) -> str`。`src/services/llm_client.py` の `BaseLLMAdapter` が吸収する。
**アダプタは `src/` に置く** — テスト側にだけ置くと本番経路にアダプタが存在せず、
テストだけが緑になる（旧実装がその状態だった）。

### Design Decision Record

| # | 論点 | 判断 | 根拠 |
|---|---|---|---|
| D-1 | LLM の受け口 | **`config["llm"]` のみ**。`llm_client=` 引数は廃止 | framework の実契約に従う呼び出し側が渡さない引数は永久に未配線になる（実際になっていた） |
| D-2 | 未注入時の扱い | **失敗（`LLM_CALL_FAILED`）**。決定論フォールバックは持たない | 会話の生成が中核能力そのもので、代替出力が存在しない。黙って別のものを返すより失敗が正しい |
| D-3 | 未注入時のリトライ | **しない**（番兵は何回呼んでも成功しない） | 指数バックオフで誤設定のデプロイに毎リクエスト数秒の空転が乗る。実測でテストスイートが 7.3 秒 → 0.27 秒 |
| D-4 | 空応答（`{"content": None}`） | **空文字**へ正規化 | `str(None)` にかけると文字列 `"None"` が回答として利用者に返る |
| D-5 | 監査イベントの `model` | **注入クライアントが名乗るもの**を優先し、無ければ config 値 | config の `model` は注入クライアントが選ぶモデルを決めない。そのまま報告すると監査ログが実際に答えたモデルと食い違う |
| D-6 | テストの注入経路 | `config["llm"]` / `resolve_llm_client()` を通す | private 属性の直接書き換えは本番の未配線を隠す（旧実装の教訓） |
| D-7 | `BaseLLM` の判定順 | **`isinstance(BaseLLM)` を duck typing より先**に置く | SDK の実クライアント（`AnthropicClient` / `AzureOpenAIClient` 等）は `__getattr__` で内部 LangChain クライアントへ委譲する。`getattr(x, "generate")` を先に見ると**バッチ用の別シグネチャの `generate`** を拾い、文字列 prompt を渡して落ちる。`__getattr__` 委譲だけの型も MRO に実体が無いことで弾く |
| D-8 | `get_output()` に `output` を追加 | **追加する**（既存キーは不変） | AGENTIC STAR runner は success 時に `output` を必須とし、無いと `returned status=success but output is missing` で `RuntimeError`（`shared/bootstrap/marketplace_app.py`）。本 AG は `formatted_output` しか返しておらず、LLM 配線を直しても AGENTIC STAR では失敗していた |
| D-9 | LLM へのメッセージ形 | **role を保ったまま渡す**（system / user / assistant） | `System: ... User: ...` を 1 つの `user` メッセージへ潰すと system 指示の優先度が失われ、履歴に混入した文言を指示と誤読しやすくなる（プロンプトインジェクション耐性の低下） |
| D-10 | 空応答 | **一時的失敗として扱う**（success にしない） | 空文字を success にすると空の assistant 発話が履歴に永続化され、利用者には「成功したが何も返らない」応答が出る |
| D-11 | リトライ対象 | 恒久エラー（`NotImplementedError` / `TypeError` / `AttributeError`）は**再試行しない** | 設定・実装・シグネチャの誤りは何回呼んでも結果が変わらず、指数バックオフの空転が乗るだけ。エラーメッセージも実試行回数から生成する |
| D-12 | バックオフの計算 | `base <= 0` を明示的に無効化 | `base ** 0` は base に関わらず 1.0 なので、`retry_backoff_base=0` を指定しても初回リトライで 1 秒眠っていた（旧コメントの「base<=1 なら実質スリープなし」は誤り。実測でテストが 5.3 秒 → 0.27 秒） |

## 6. context window overflow ポリシー

長時間会話で履歴が無制限に増えると LLM の context を溢れさせるため、`HistoryManagerNode` で制御：

1. ロード済み履歴に今回の user turn を append
2. `config.history_window_turns`（デフォルト 10）で **直近 N ターンを上限**に切り詰め（古いターンから drop）
3. **さらに `config.history_token_budget`（デフォルト 3000）でトークン予算による切り詰め**を適用（ターン数より優先される実基準）。`system_prompt` は予算対象外で常に先頭固定
4. トークン数の見積もりは概算（文字数ベースの軽量推定で可。厳密 tokenizer は不要）

> **設計判断（2026-06-01 確定）:** MVP は「直近 N ターン上限 + トークン予算切り詰め」。
> ターン数だけでなくトークン予算を併用することで、長文ターンが多くてもモデル上限に正確に収まり溢れ事故を防ぐ。
> 要約ベースの圧縮（rolling summary）は複雑度が上がるため **Sprint 2** の拡張点とする
> （過去ターンの意味検索は RAG=VectorRAGAgent の責務であり、純粋 ChatAgent では採用しない）。

**保存側の肥大化対策（#3, サーベイ 2026-06-02）:** 上記の window は「LLM 入力」を絞るだけで、
`ConversationRepository` には全ターンが蓄積し無制限に増えうる。業界標準（rolling window cap /
TTL eviction / ハイブリッド）に倣い、**保存側にも rolling window cap を実装**:
`config.history_max_stored_messages`（デフォルト 50）で `save()` 時に直近 N 件に切り詰める。
TTL/eviction と要約 consolidation は本番永続 IF / Sprint 2 で扱う。

---

## 7. マルチセッション分離（履歴漏洩防止）

ChatAgent の最重要セキュリティ要件。顧客 A の履歴が顧客 B の応答に混入してはならない。

- `conversation_history` のロード／セーブは **必ず `session_id` を key** に行う（`ConversationRepository.load(session_id)` / `.save(session_id, ...)`）
- `session_id` が空 / 未指定の場合は `state["error"]` を立てて early-exit（`error_code = MISSING_SESSION_ID`。履歴を一切ロードしない）
- State の `conversation_history` は当該 `session_id` 分のみを保持。複数セッションを 1 つの State に混在させない
- PB テストで「session A の履歴が session B の windowed_messages に出現しない」ことを検証（後述 PB-7）

---

## 8. セキュリティ設計

### S-1 — Trust チェック（`required_trust_level` / `BaseNode.__call__` 自動強制）
- ADR-006 に基づき S-1 は **per-node 自動強制**に統一。各ノードは
  `required_trust_level = TrustLevel.VERIFIED_EXTERNAL` を ClassVar 宣言し、`BaseNode.__call__` が
  `state["caller_trust_level"]` と比較して不足を拒否する。ノードは S-1 を手動実装しない（master
  C1-002・C1-021 と同一パターン）。agent 側 `__pre_invoke__` は持たない。
- **production エントリは `invoke()`**（`run()` は本番に存在しない）。`invoke()` が
  `config["configurable"]["invocation_context"]` の `caller_trust_level`（無ければ `trust_level` alias）を
  `state["caller_trust_level"]`（**int = checkpoint 安全な primitive**）として書き込む。InvocationContext を
  state に持たない（state-safety / SEC-C1003-001）。
- **absent context = ANONYMOUS の default-deny**（invoke() の baseline = ANONYMOUS）。
- `caller_trust_level < required_trust_level` なら `state["error"]` + `error_code = S1_TRUST_DENIED` を立て、
  `_invoke_impl` のノードループが break（後続ノード未実行）。直接ノード呼び出し（`node(state)`）でも同等に強制。

### S-2 — 入力ゲート（`_security_gate_input`）
- `user_input` の credential pattern scan（`api_key|token|secret|password=...` / `Bearer ...` / `sk-...` / `BEGIN PRIVATE KEY`）+ prompt injection の基本検出
- 検出時: `state["error"]` を立てる（`error_code = S2_INJECTION_DETECTED`。`SessionInitNode` 内で先行実施）
- **ロールなりすまし緩和:** `LLMInferenceNode` がプロンプト構築時にユーザー文中の行頭 role マーカー（`System:`/`User:`/`Assistant:`）を全角化して無害化（framework の `LLMClient.generate(str)` が文字列 IF のため。`src/masking.py` ではなく `llm_inference._sanitize`）

### S-3 — 出力ゲート（`_security_gate_output`）— **実装済み**
- `ResponseNode` / `ChatAgent._security_gate_output` で出力を処理。**2 段構え:**
  1. **PII マスク（redact して通過）:** `config.mask_patterns` の既知ラベル（`src/masking.py` の `MASK_REGISTRY`: `mynumber` 12桁 / `account_number` 文脈付き）にマッチした箇所を `[REDACTED:<label>]` に置換。マスク後テキストを `formatted_output` とし、**永続化もマスク後**（PII を保存しない）
  2. **credential ブロック（全停止）:** マスク後に credential（sk-/JWT 等）が残れば `state["error"]`（`error_code = S3_BLOCKED`）で応答ブロック（永続化も返却もしない）
- **mask 方針:** 標準 ChatAgent は PII を網羅列挙しない。`mask_patterns` は最小・拡張可能リストで、未登録ラベル（`appi_sensitive` 等）は派生が `MASK_REGISTRY` に登録する拡張ポイント。クレジットカード番号等も派生で追加

### S-4 — 監査トレース（`_emit_trace_event`）
- ログ対象（フィールド名・件数・メタのみ、発話本文は出さない）:
  - `template_id` / `session_id` / `history_turn_count` / `model`
- `user_input` / `llm_response` の **本文は絶対に記録しない**

### S-5 — 認証情報検出（自動）
- `__init_subclass__` が import 時に JWT / API キーリテラルをスキャン（framework 提供）

---

## 9. エラー伝播戦略

各ノードは異常時に **`state["error"]`（真値のメッセージ）を立てて** 早期 exit する。
`AgentBaseGraph.invoke()` 内の **`_invoke_impl` ノードループは `state.get("error")` が立つと後続ノードを
実行せず break** する。具体的なコードは `error_code` / 詳細は `error_message` に格納する。
`ResponseNode` は終端で **必ず `formatted_output` を emit**（エラー時はエラーメッセージを安全な形で返す）。

> **重要:** 早期終了のトリガーは `state["error"]`。`error_code` だけを立てても `_invoke_impl` は break しないため、
> 必ず `state["error"]` をセットすること（`error_code` は分類用の付随情報）。

### エラーコード一覧

| コード | 発生源 | 意味 |
|---|---|---|
| `EMPTY_INPUT` | SessionInitNode | `user_input` が空 / 欠落 |
| `MISSING_SESSION_ID` | SessionInitNode | `session_id` 未指定（履歴ロード不可） |
| `HISTORY_LOAD_FAILED` | SessionInitNode | 履歴ストアからのロード失敗 |
| `LLM_CALL_FAILED` | LLMInferenceNode | LLM 呼び出し失敗（リトライ尽き） |
| `S1_TRUST_DENIED` | `BaseNode.__call__`（各ノード `required_trust_level`） | Trust レベル不足（caller_trust_level < required） |
| `S2_INJECTION_DETECTED` | `_security_gate_input` | 入力に credential / injection |
| `S3_BLOCKED` | `_security_gate_output` | 出力に credential / PII |
| `INTERNAL_ERROR` | 任意 | 予期せぬ例外 |

### Graceful degradation
- 履歴ロード失敗 → 履歴なし（新規会話扱い）で続行し、`HISTORY_LOAD_FAILED` を warning ログに記録（会話自体は継続）
- LLM 呼び出し失敗 → `max_retry` 回まで指数バックオフ（`retry_backoff_base ^ attempt` 秒）で再試行（**実装済み** `LLMInferenceNode`）。尽きたら `state["error"]`（`error_code = LLM_CALL_FAILED`）を立て、`_invoke_impl` のノードループが break して安全に終了

---

## 10. 派生テンプレートのオーバーライド面

> 会話能力を再利用する下流テンプレート（system_prompt 差分だけの設定インスタンス＝ Cat 1、
> または会話に複数ステップを足して特定の仕事を完遂するもの＝ Cat 2）が override する面。

| Override 対象 | 派生での典型例 |
|---|---|
| `config.system_prompt_template` | HCR: 臨床トリアージ用の問診プロンプト |
| `config.history_window_turns` | 長い診断会話は 20 ターン、短い FAQ は 4 ターン |
| `config.model` / `temperature` | GOV: 厳格回答のため `temperature: 0.0` |
| `config.required_trust_level` | 社内ヘルプデスク派生は `INTERNAL` に昇格 |
| `LLMInferenceNode`（軽微 override） | ドメイン固有の後処理が必要な場合のみ |

**Override 禁止項目**: `s3_gate_enabled`（常に true）/ `mask_patterns` の `appi_sensitive` /
マルチセッション分離ロジック（`session_id` を key とする履歴 I/O）。

---

## 11. 呼び出しモード

### Top-level invocation（`run` で確定 — SummarizationAgent リリース版に準拠）

```python
agent = ChatAgent(config=yaml_cfg)
state = {"user_input": "残高照会の方法を教えて", "session_id": "sess-abc-123"}
config = {"configurable": {"invocation_context": ctx}}   # ctx.caller_trust_level >= VERIFIED_EXTERNAL
result = agent.invoke(state, config)                      # production エントリ（AgentBaseGraph から継承）
print(result["formatted_output"])
```

> **解決済み（旧 Open Item「run vs invoke」）:** production 実行は `AgentBaseGraph` 継承の **`invoke(state, config)`**
> を使う（`run()` は本番に存在せず使用禁止）。**独自 `run()` / `invoke()` / `self.graph`
> 合成は定義しない**（CR-R1-01 で合成は NG 判定）。`invoke()` は `caller_trust_level` を state に書き込み、
> `_invoke_impl` が各ノードを `__call__` 経由（S-1 自動強制）→ `execute(state)` を entry から順に呼び、
> `state["error"]` が立つと break する。
>
> ⚠️ **ノード契約（ノード契約）:** 実 Framework の `BaseNode` の抽象メソッドは
> **`execute()` のみ**。ノードに `_invoke_impl()` を実装してはいけない（CI の
> `test_execute_method_signature()` が FAIL する）。各ノードは `execute(self, state) -> dict` のみを実装する
> （SDK v1.0.0 の `BaseNode.__call__` は `execute(state)` を state 1 引数のみで呼ぶ。`config` は渡されないため取らない）。

---

## 12. テスト計画（概要 — 詳細は `docs/03_test_spec.md` で起案）

### Unit Tests（目標 8+ TC）
- 各ノード（4 個）の正常系・異常系
- 履歴ウィンドウ切り詰め（N ターン超過時）
- 空入力 / session_id 欠落のエラー伝播

### Integration Tests
- `agent.invoke()` E2E（多ターン会話の継続）
- マルチセッション分離（session A → session B で履歴が混入しない）

### Proof-of-Boundary Tests（目標 6+ PB）
- PB-1: BaseNode → EventEmitter（S-4 trace 必発火）
- PB-2: State serialization（primitives のみ、Pydantic 混入なし）
- PB-3: LLMInferenceNode → 外部 LLM service（実呼び出し境界）
- PB-4: Import isolation（src/ が agenticstar を import しない）
- PB-5: Checkpoint safety（履歴 / credential 混入なし）
- PB-6: Invoke lifecycle order（pre_invoke → gate_input → nodes → gate_output → trace）
- **PB-7（本テンプレ固有）: マルチセッション履歴漏洩**（session A の履歴が session B の windowed_messages に出現しない）

---

## 13. Definition of Done（設計 → 実装の移行条件）

- [x] `docs/02_design.md` 起案（本ドキュメント）
- [x] **L1 Base type 決定: `AgentBaseGraph`（L1 直接継承で確定）** — v0.16 §5-1 / AgentCore SDK v1.0.0 準拠。旧 Level 2 base ChatAgent は概念パターンとして参照のみ（継承ターゲットではない）
- [x] 4 ノード責務明確化
- [x] State スキーマ確定
- [x] Config 仕様確定
- [x] セキュリティモデル明示（S-1〜S-5 全層）
- [x] context window overflow ポリシー明示
- [x] マルチセッション分離設計明示
- [x] 派生テンプレート override ガイドライン同梱
- [ ] 設計レビュー承認（設計 → 実装の移行時に実施）

---

## 14. Definition / 設計判断サマリ

| 判断 | 決定 |
|---|---|
| 継承 | **Agent が Graph を継承**: `ChatAgent(ChatGraph)` / `ChatGraph(AgentBaseGraph)`（v0.16 L1 直接 + SummarizationAgent CR-R1-01） |
| 実行 | production エントリは `AgentBaseGraph` 継承の **`invoke(state, config)`**（`run()` は本番不在・使用禁止。独自 run / invoke / self.graph 合成も不可） |
| S-1 強制 | 各ノードが `required_trust_level` を宣言し `BaseNode.__call__` が `state["caller_trust_level"]` で自動強制（ADR-006） |
| エラー伝播 | `state["error"]` で早期終了（`_invoke_impl` のノードループが break）。`error_code`/`error_message` は付随情報 |
| ノード数 | 4 固定（SessionInit / HistoryManager / LLMInference / Response） |
| 履歴管理 | 直近 N ターン上限 + **トークン予算切り詰め**（要約圧縮は Sprint 2） |
| セッション分離 | `session_id` を key とする履歴 I/O、混在禁止 |
| Trust | **`TrustLevel.VERIFIED_EXTERNAL`（=1）**（ANONYMOUS 拒否・absent=default-deny。派生で INTERNAL に昇格可） |
| 検索/ツール | なし（純粋会話に限定、Single-responsibility） |

---

## 15. 既知の未解決事項（Open Items）

| 項目 | 必要な判断 | オーナー | 解決方針 |
|---|---|---|---|
| ~~実行メソッド名（`run` vs `invoke`）~~ | — | — | ✅ **解決**: production エントリは **`invoke()`** で確定（`run()` は本番不在・使用禁止。2026-06-04 のレビュー指摘で是正） |
| ~~Trust レベル既定値~~ | — | — | ✅ **解決**: `TrustLevel.VERIFIED_EXTERNAL`（=1）で確定（2026-06-02 のレビュー指摘で是正）。本番 enum は ANONYMOUS/VERIFIED_EXTERNAL/INTERNAL の 3 値で、旧記載の AUTHENTICATED は不在（import で AttributeError）だったため是正 |
| ~~context 切り詰め方式~~ | — | — | ✅ **解決**: 直近 N ターン上限 + トークン予算（2026-06-01 確定）。rolling summary は Sprint 2 |
| `ConversationRepository` IF | 履歴永続化の platform 提供 IF 仕様 | platform | **MVP は付録 A の in-memory stub IF で固定**。本番 IF が異なる場合の差分吸収は付録 A の方針に従う |
| ~~node `execute()` の `config` 引数~~ | — | — | ✅ **解決（SDK v1.0.0 移行）**: `execute(self, state)`（config なし）で確定。1.0.0 の `BaseNode.__call__` は `execute(state)` を state 1 引数のみで呼び、`config` を渡さないため、`config=None` は未使用の余計な引数になる。trust は `state["caller_trust_level"]` から取得（ADR-006）。3 ノードから `config` 引数を削除し PB `test_execute_method_signature` を `(self, state)` 契約に更新（移行 Issue #15） |

> **実装ブロッカー回避（指摘2）:** `SessionInitNode`（`.load()`）と `ResponseNode`（`.save()`）が
> 本 IF に依存するため、**MVP の IF を付録 A で凍結**する。本番 platform IF が将来変わっても、
> 改修を `ConversationRepository` 実装クラス 1 箇所に閉じ込められるよう、ノードは付録 A のシグネチャのみに依存する
> （ノードが repository 内部実装に直接触れない）。IF 変更リスクは MVP として受容する。

---

## 付録 A. ConversationRepository MVP スタブ IF（凍結）

履歴永続化の本番 platform IF は未確定（§15）だが、実装が止まらないよう **MVP のシグネチャを以下に凍結**する。
`sample/chat_agent` の `ConversationRepository` と同形。`SessionInitNode` / `ResponseNode` は
**このシグネチャのみに依存**し、内部実装（in-memory / DB / platform IF）には触れない。

```python
class ConversationRepository:
    """会話履歴の永続化（MVP は in-memory 実装）。session_id を分離キーとする。"""

    def load(self, session_id: str) -> list[dict]:
        """session_id に紐づく会話履歴を返す。
        返り値: [{"role": "user"|"assistant", "content": str}, ...]（JSON-serializable のみ）。
        未知 session_id は [] を返す（新規会話扱い、例外を投げない）。"""

    def save(self, session_id: str, messages: list[dict]) -> None:
        """session_id に会話履歴を保存（上書き or 追記は実装定義。MVP は上書き）。
        messages は load() と同形。PII を含みうるため本文はログに出さない。"""
```

**契約・スコープ:**
- **分離キーは必ず `session_id`**（§7 マルチセッション分離）。他セッションのデータを返さない／混ぜない
- 返り値・引数は **primitives + JSON-serializable のみ**（State 安全則 §4。Pydantic/dataclass 禁止）
- MVP 実装は **プロセス内 in-memory dict**（`{session_id: list[dict]}`）。永続化なし（再起動で消える前提）
- 本番 platform IF（永続ストア）への差し替えは `ConversationRepository` 実装クラスの内部のみで完結させる
  （ノードはこの IF に対してのみコーディングする）。IF が変わってもノード改修は不要

---

## 16. 変更履歴

| 日付 | バージョン | 作成者 | 変更内容 |
|---|---|---|---|
| 2026-06-01 | 0.1（draft） | 開発者AAA | impl #1 で起案。L1 直接継承、4 ノード固定、直近 N ターン履歴ウィンドウ、マルチセッション分離を確定。AlertNotificationAgent の構造を参考 |
| 2026-06-01 | 0.2（draft） | 開発者AAA | Cat 定義改訂（2026-06-01）に対応。Cat 1 の根拠を「業種非依存だから」→「単一の技術能力＝多ターン会話を汎用提供するから」に修正。Industry 軸と Cat 軸の独立を明記。CMN-C1-003 は改訂後も Cat 1 で確定（ID 不変、リナンバリングは品質管理プロセス側の判断）。あわせて旧前提「Cat 2 派生＝config 差分で作る」を全面是正し、「system_prompt 差分だけの設定インスタンス＝ Cat 1 / 複数ステップで仕事を完遂＝ Cat 2」と派生区分を明確化 |
| 2026-06-02 | 0.3（draft） | 開発者AAA | SummarizationAgent（リリース版）+ 旧スタブ環境を実地調査し Open Items を解決。(1) 実行は `run()` で確定（invoke / 独自 run / self.graph 合成は不可、CR-R1-01）。(2) **Agent が Graph を継承**する 2 段構成（`ChatAgent(ChatGraph(AgentBaseGraph))`）に §2 修正。(3) 早期終了は `state["error"]`（run() が break）に統一。(4) Trust を `VERIFIED_EXTERNAL`（enum 不在）→ **`TrustLevel.AUTHENTICATED`** に修正 + absent=default-deny。(5) context 切り詰めを直近 N ターン + **トークン予算** に。(6) mask_patterns は最小・拡張可能、シークレット類は framework credential scan が担保、と §5/§8 に明記 |
| 2026-06-02 | 0.4（draft） | 開発者AAA | レビュー指摘 2 件を反映。[指摘1] §11 の「run() が `_invoke_impl()` を呼ぶ」を **`execute()`** に統一し、ノード契約 注記を追加（ノードに `_invoke_impl()` を実装すると `test_execute_method_signature()` が FAIL）。[指摘2] `ConversationRepository` の MVP スタブ IF を **付録 A で凍結**（`load(session_id)->list[dict]` / `save(session_id, messages)->None`）。ノードは本 IF のみに依存し IF 変更を実装クラス内に閉じ込める方針を §15 に明記 |
| 2026-06-03 | 0.6（draft） | 開発者AAA | アーキテクチャレビュー指摘（HIGH 1 件 / ADVISORY 1 件）を是正。本番 `TrustLevel` enum は `ANONYMOUS=0 / VERIFIED_EXTERNAL=1 / INTERNAL=2` の 3 値で、`AUTHENTICATED`（不在＝import 時 AttributeError）・`VERIFIED_INTERNAL`（旧エイリアス）は使用禁止。Trust 最小要件を **`AUTHENTICATED`（=1）→ `VERIFIED_EXTERNAL`（=1）** に修正（src/agent.py・config/agent.yaml・src/nodes/session_init.py・tests/conftest.py スタブ・テスト・本 design doc）。数値（=1）は不変のため S-1 ゲート挙動は等価。0.3 で「VERIFIED_EXTERNAL は enum 不在」とした前提が誤りであったことを是正 |
| 2026-06-02 | 0.5（draft） | 開発者AAA | 実装レビューの gap を解消。(#1) `mask_patterns` を **実適用**（`src/masking.py` MASK_REGISTRY、mynumber/account_number を redact、未登録ラベルは拡張ポイント）— S-3 を「PII マスク通過 + credential ブロック」の 2 段に。(#2) **LLM リトライ**（max_retry + 指数バックオフ retry_backoff_base、尽きたら LLM_CALL_FAILED）を実装。(#3) サーベイに基づき保存側 **rolling window cap**（history_max_stored_messages=50）を `ConversationRepository.save()` に実装。(#6) ロールなりすまし緩和（ユーザー文中 role マーカーの全角無害化）を追加。テスト 41 件 PASS |
| 2026-06-04 | 0.7（draft） | 開発者AAA | アーキテクチャ / セキュリティレビュー指摘（HIGH 2 件）を是正。(1) `src/agent.py` の dead override `__pre_invoke__()`（本番 framework が dispatch しない）を削除。S-1 は ADR-006 どおり **entry node = SessionInitNode** で強制（バイパス不可）に一本化し、docstring / §3 / §8 / エラーコード表を実装に整合。(2) `ChatState` から `invocation_context: Optional[object]`（msgpack/checkpoint 不安全・state-safety 違反）を削除し、`SessionInitNode` の `invocation_context` 取得を **config 専用**化（state fallback 撤去、出力 strip は多層防御として残置）。挙動等価（S-1 拒否は従来どおり `S1_TRUST_DENIED`）。`docs/07_operation_guide.md` の S-1 行も SessionInitNode に修正。テスト緑維持 |
| 2026-06-04 | 0.8（draft） | 開発者AAA | レビュー指摘（MEDIUM run/invoke + MINOR trust 属性名）を受け、S-1 を **ADR-006 canonical 自動強制**へ統一（C1-002・C1-021 と同一パターン）。(1) production エントリを `invoke()` に明確化（`run()` は本番不在・使用禁止）。framework stub（`tests/conftest.py`）を `invoke()`＝caller_trust_level 書込→S-2→`_invoke_impl`（各ノード `__call__` が S-1）→S-3→S-4 に刷新、`run()` / `__pre_invoke__` を撤去。(2) 各ノードが `required_trust_level = VERIFIED_EXTERNAL` を ClassVar 宣言し、`BaseNode.__call__` が `state["caller_trust_level"]`（int・checkpoint 安全）で自動強制。`SessionInitNode` の手動 trust 読み取りを撤去。(3) trust 属性名の不一致リスク（`trust_level` vs `caller_trust_level`）を `InvocationContext` の両対応 + `caller_trust_level` 基準で解消。テスト 42 件 PASS（S-1 拒否は `node(state)` 経路で検証、PB-6 ライフサイクルを `in→impl→out→trace` に更新） |
