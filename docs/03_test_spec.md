# CMN-C1-003 ChatAgent — テスト仕様書

**Template ID:** CMN-C1-003
**対象:** テストフェーズ（docs/02_design.md §12 を具体化）
**作成日:** 2026-06-02
**テスト方式:** pytest（`tests/conftest.py` が framework.* スタブを登録）
**実行コマンド:** `python -m pytest tests/ -v` / `python -m pytest tests/proof_of_boundary/ -v`
**最終結果:** ✅ 全 41 件 PASS（unit 26 / integration 6 / proof_of_boundary 11 ※parametrize 含む）

---

## 1. テスト構成

```
tests/
├── conftest.py                       framework.* スタブ登録（BaseNode/AgentBaseGraph/TrustLevel 等）
├── unit/
│   ├── test_session_init.py          SessionInitNode（S-1/S-2/検証/履歴ロード）
│   ├── test_history_manager.py       HistoryManagerNode（window 上限 + トークン予算）
│   ├── test_llm_inference.py         LLMInferenceNode（単一 LLM 呼び出し）
│   └── test_response.py              ResponseNode（S-3 + 履歴永続化）
├── integration/
│   └── test_chat_agent.py            E2E（多ターン・マルチセッション分離・invoke ライフサイクル）
└── proof_of_boundary/
    └── test_pb_placeholder.py        PB-1..7 + ノード契約
```

---

## 2. Framework Compliance TC（必須項目）

| TC-ID | 試験内容 | 期待結果 | 対応テスト |
|---|---|---|---|
| TC-01 | State contract（flat / 非スカラ混入なし） | 出力 state は str/int/bool/None のみ | `test_pb2_state_values_are_primitives` |
| TC-02 | S-1 trust 拒否（ADR-006 自動強制） | absent context → `state["error_code"] = S1_TRUST_DENIED`（例外ではなく `_invoke_impl` break） | `test_invoke_denies_absent_context` |
| TC-03 | JWT/Credential 検出 | 入力/出力の credential を検出しブロック | `test_s2_credential_in_input_rejected` / `test_s3_blocks_credential_in_output` |
| TC-04 | InvocationContext は configurable 経由のみ | state に context を残さない | `test_invocation_context_not_leaked_into_output` / `test_pb5_checkpoint_safety` |
| TC-05 | AuditSink（trace）記録 | 全 invoke で trace 発火・本文なし | `test_pb1_trace_event_fires` / `test_invoke_lifecycle_success_emits_trace` |
| TC-06 | `_security_gate_input` 非バイパス | PII/credential ゲート常時実行 | `test_s2_credential_in_input_rejected` |
| TC-07 | `_security_gate_output` 非バイパス | 出力安全フィルタ常時実行 | `test_s3_blocks_credential_in_output` |
| TC-08 | `required_trust_level` 宣言・強制 | trust 不足 → 拒否 | `test_s1_absent_context_is_denied` / `test_s1_anonymous_is_denied` |

## 3. 機能 TC（ChatAgent 固有）

| TC-ID | 試験内容 | 期待結果 | 対応テスト |
|---|---|---|---|
| TC-09 | 空入力拒否 | `EMPTY_INPUT` | `test_empty_input_rejected` |
| TC-10 | session_id 欠落拒否 | `MISSING_SESSION_ID` | `test_missing_session_id_rejected` |
| TC-11 | 履歴ロード・system_prompt 解決 | 正常系で履歴・prompt 反映 | `test_happy_path_loads_and_resolves` |
| TC-12 | 履歴 window 上限（直近 N ターン） | N 件に切り詰め・最新 turn 残存 | `test_turn_cap_keeps_most_recent` |
| TC-13 | トークン予算切り詰め | 予算超過は最古から drop | `test_token_budget_trims_oldest` |
| TC-14 | 単一 LLM 呼び出し | `llm_response` 生成 | `test_generates_llm_response` |
| TC-15 | 多ターン履歴蓄積 | turn2 に turn1 が含まれる | `test_multi_turn_history_accumulates` |
| TC-16 | 履歴永続化（full） | loaded + user + assistant を保存 | `test_happy_path_persists_full_history` |
| TC-17 | invoke ライフサイクル正常系 | output + trace | `test_invoke_single_turn_produces_output` |
| TC-18 | S-3 PII マスク適用 | mynumber を `[REDACTED]` に置換し通過・保存もマスク後 | `test_s3_masks_pii_in_output` |
| TC-19 | LLM リトライ成功 | 一時失敗後にリトライ成功 | `test_retry_then_success` |
| TC-20 | LLM リトライ枯渇 | 全失敗で `LLM_CALL_FAILED` | `test_retry_exhausted_sets_llm_call_failed` |
| TC-21 | role マーカー無害化（#6） | ユーザー文の `Assistant:` を全角化 | `test_role_marker_sanitized_in_prompt` |
| TC-22 | 保存 rolling window cap（#3） | 直近 N 件のみ保存 | `test_rolling_window_cap_keeps_most_recent` |
| TC-23 | リポジトリ分離・コピー返却 | session 分離 / load はコピー | `test_session_isolation` / `test_load_returns_copy` |

---

## 4. Proof-of-Boundary（docs/02_design.md §12）

| PB-ID | 境界 | 期待結果 | 対応テスト |
|---|---|---|---|
| PB-1 | BaseNode → EventEmitter（S-4） | trace 必発火 | `test_pb1_trace_event_fires` |
| PB-2 | State serialization | primitives のみ | `test_pb2_state_values_are_primitives` |
| PB-3 | LLMInferenceNode → 外部 LLM | 文字列応答を返す | `test_pb3_llm_service_boundary_returns_text` |
| PB-4 | Import isolation | src/ が agenticstar を import しない（AST scan） | `test_pb4_import_isolation_no_agenticstar` |
| PB-5 | Checkpoint safety | context/credential 混入なし | `test_pb5_checkpoint_safety` |
| PB-6 | Invoke lifecycle order | in→impl→out→trace（`invoke()`。run()/__pre_invoke__ は不在） | `test_pb6_invoke_lifecycle_order` |
| PB-7 | **マルチセッション履歴漏洩**（固有） | session A の履歴が B に出現しない | `test_pb7_multi_session_no_leak` |
| 契約 | ノード契約（ノード契約） | 各ノードは execute() のみ・`_invoke_impl` 不実装 | `test_node_contract_execute_only`（4 ノード parametrize） |

---

## 5. 実行結果

```
$ python -m pytest tests/ -q
41 passed
$ python -m pytest tests/proof_of_boundary/ -q
11 passed
```

全 TC / PB が PASS。CI（run-tests ジョブ）で再現される。

---

## 6. 変更履歴

| 日付 | 版 | 作成者 | 内容 |
|---|---|---|---|
| 2026-06-02 | 0.1 | 開発者AAA | impl #8。TC-01〜17 + PB-1〜7 + ノード契約テストを起案・全 PASS 確認 |
| 2026-06-02 | 0.2 | 開発者AAA | gap 解消に伴い TC-18〜23 追加（mask / retry / role 無害化 / 保存 cap / リポジトリ）。全 41 件 PASS |

## 実 LLM 1 回転検証

モックだけで固めた実装は実 LLM を繋いだ瞬間に動かないことがあるため、注入シームへ実 LLM クライアントを刺して 1 回 invoke し、**中身のある出力が出ること**を確認している（`gitproject-dev-flow.md` §14-1-bis）。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-28 |
| 使用 LLM | ローカル CLI（Azure OpenAI）を `BaseLLM` 実装として注入 |
| 注入シーム | `live_llm_injector` |
| 実 LLM 呼び出し | 1 回 |
| 実行で判明した欠陥 | 本 AG は `generation_mode: "llm"` を宣言しながら、外側 Graph の独自引数 `llm_client=` でしか LLM を受け取れず、`src/api/server.py` は `ChatAgent(config=...)` としか呼ばないため**本番は毎回 `LLM_CALL_FAILED`** だった。テストが緑だったのは conftest が private 属性を直接書き換えていたため。`config["llm"]` 経由へ是正し、併せて空応答が文字列 `"None"` になる欠陥・監査ログのモデル名不一致・未設定クライアントの空転リトライを修正 |
| 出力 | ['<text output: 32字>'] が非空。縮退理由なし |

本番の LLM は `AnthropicClient` 等で、これは検証用シーム。テンプレの `src/` は変更していない（一時テストを生成→実行→削除する方式）。CI には入れていない（ローカル CLI と鍵が要るため）。

