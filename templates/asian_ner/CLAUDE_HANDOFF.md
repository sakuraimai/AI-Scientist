# Claude Handoff Memo — asian_ner AI Scientist (Sakana Screening)

**作成日:** 2026-07-15  
**目的:** Cursor Enterprise 終了後、Claude へ作業を引き継ぐための参照メモリ  
**リポジトリ:** `sakuraimai/AI-Scientist`  
**テンプレート:** `templates/asian_ner/`

---

## 1. プロジェクト概要

### タスク
- 29言語 WikiAnn NER（`asian_east_sea` 28言語 + `ru`）で XLM-R 転移学習
- **Imai et al. (ACL 2023 SRW)** 延長: linguistic typology がクラスタリング転移を改善するか検証
- **6固定条件**（名前変更禁止）:
  - `linguistic_clustering`, `embedding_clustering`, `per_language`, `all_mixed`, `matched_random_embedding`, `matched_random_linguistic`

### AI 成功基準（`prompt.json`）
- `embedding_clustering` が `matched_random_embedding` を `low_resource_macro_f1` および/または `average_f1` で上回る
- 同様に linguistic ペアも

### 主評価指標
- **Primary:** `low_resource_macro_f1`（train≤500 の言語群の macro F1）
- **Secondary:** `average_f1`, `mongolian_f1`（illustrative のみ）

### 公平 floor プロトコル（`cap=0`, `downsample_to=target_cluster`）
- embedding/linguistic クラスタ: ターゲットクラスタの **フル WikiAnn**（ダウンサンプルなし）
- `matched_random_embedding`: embedding ターゲットクラスタと **同じ N** を 29言語プールからランダムサンプル
- `matched_random_linguistic`: linguistic クラスタと同じ N
- `all_mixed`: フルプール ceiling（floor run では **skip**）
- デフォルト `--skip_conditions all_mixed`（Pod/Mac ともに floor 用）

---

## 2. 実験タイムライン

| 段階 | 実施者 | 内容 | 結果 |
|------|--------|------|------|
| Human floor | 人間 | fair cap=0 6条件（`run_0`） | matched >> clustering |
| AI run #1 | AI Scientist Pod | `head_final_cohesion`（5 run） | 未達・微改善のみ |
| Idea gen | Mac LLM | 11案 + S2 novelty | 全件 `novel: true` |
| AI run #2 | AI Scientist Pod | `typology_weighted_mixture`（5 run） | 未達・悪化 |
| Writeup | Mac | 両 run の PDF 手動生成 | 完了 |

---

## 3. 人間 floor 結果（`run_0`）

**パス:** `templates/asian_ner/run_0/`（Pod: `run_asian_cap0_target_cluster`）  
**Git:** `main` ブランチ `85cec82`

| 条件 | low_resource_macro_f1 | average_f1 |
|------|----------------------|------------|
| embedding_clustering | **0.440** | 0.469 |
| matched_random_embedding | **0.615** | 0.667 |
| linguistic_clustering | 0.439 | 0.467 |
| matched_random_linguistic | 0.587 | 0.649 |

**学習構成（floor）:**
- embedding クラスタ: **{ja, ko, mn}**, N≈**40,100**
- matched_random_embedding: 同 N を 29言語からサンプル

**結論:** 公平 matched control 下でも **matched random が大幅に勝利** → 固定 N 下では「言語多様性」が typology 同質クラスタ学習に勝つ

---

## 4. AI run #1 — `head_final_cohesion`

### 設定
- Pod, `gpt-4o-2024-05-13`, `--skip-idea-generation --skip-novelty-check`
- 単一 idea: ja-ko-mn を同一クラスタに強制（cohesion）
- 結果 dir: `results/asian_ner/20260712_123632_head_final_cohesion/`
- Git ブランチ: `ai-head-final-cohesion-results`（`21ed444` に tar）

### 結果

| Run | 変更 | emb low_res | matched_emb low_res |
|-----|------|-------------|---------------------|
| run_0 | baseline コピー | 0.440 | 0.615 |
| run_1 | hard ja-ko-mn constraint | **0.449** | 0.658 |
| run_2–5 | `HYBRID_TYPOLOGY_PENALTY` 1→10 sweep | 0.430 | 0.625（全同一） |

- **Success: False**（writeup: Pod で `algorithm.sty` 不足 → Mac で PDF 生成）
- run_2–5 同一 = penalty sweep 無効

### 成果物（Mac）
- `results/asian_ner/20260712_123632_head_final_cohesion/`（tar 展開済み）
- PDF: 前回 Mac で `plot.py` + `latex/template.tex` 修正後に生成

### 教訓
- cohesion 強化だけでは matched に届かない
- `prompt.json` に **AI RUN FAILURE** セクション追記済み（Mac 未コミットの可能性あり）

---

## 5. アイデア生成（11案）

**パス:** `templates/asian_ner/ideas.json`  
**生成:** seed 6 + LLM 5 = 11（`max_num_generations=5`）  
**コミット:** `fb0ccff`（`ai-typology-weighted-mixture-results` ブランチ）

### 出所内訳

**人間 seed（6）** — `seed_ideas.json`:
1. `target_cluster_downsample_budget` — プロトコル説明（実施済み）
2. `head_final_cohesion` — AI run #1 実施済み
3. `script_aware_mn_ru_penalty` — **未実施・推奨候補**
4. `typology_weighted_mixture` — AI run #2 実施済み
5. `k2_silhouette_head_final` — 未実施
6. `low_resource_reweight` — 未実施・推奨候補

**LLM 新規（5）:**
7. `labse_embedding_clustering` — スコープ大（LaBSE 導入）
8. `dynamic_cluster_sampling` — 未実施
9. `progressive_finetuning` — スコープ大
10. `meta_learning_multilingual` — スコープ外
11. `multitask_auxiliary` — スコープ外

### IFN スコア
- **I** = Interestingness, **F** = Feasibility, **N** = Novelty（1–10、LLM 自己評価）
- **選択に IFN は使われない** — novelty フィルタ + `ideas.json` 順 + 人間が `--idea-names` で指定

### AI Scientist の idea 選択ロジック
1. `seed_ideas.json` 全件 + LLM 生成を `ideas.json` に保存
2. `check_idea_novelty()` → `novel: true/false`（S2/OpenAlex）
3. `novel_ideas` を **ファイル順に全件** 実行（デフォルト）
4. **`--idea-names typology_weighted_mixture`** で1本に絞る（`launch_scientist.py` に追加、`fb0ccff`）

---

## 6. AI run #2 — `typology_weighted_mixture`

### 設定
- Pod, 同上 + `--idea-names typology_weighted_mixture`
- 結果 dir: `results/asian_ner/20260714_220439_typology_weighted_mixture/`
- Git: `typology_weighted_mixture_results.tar.gz` on `ai-typology-weighted-mixture-results`（`40aef99`）

### F1 結果

| Run | emb low_res | matched_emb low_res | 判定 |
|-----|-------------|---------------------|------|
| run_0 | 0.440 | 0.615 | 負け |
| run_1–5 | **0.349** | 0.589 | 負け（run_1–5 全同一） |

### トレーニングサンプル数（`detailed_results.json` より）

| Run | embedding N | embedding langs | matched_emb N | matched langs |
|-----|-------------|-----------------|---------------|---------------|
| run_0 | （旧形式） | ja, ko, mn | — | — |
| run_1–5 | **20,100** | **mn, ko のみ** | **20,100** | 29言語 |

**重要 — プロトコル逸脱の疑い:**
- 人間 floor: embedding **N≈40,100・{ja,ko,mn}**
- AI run #2: embedding **N=20,100・{mn,ko}**（ja 脱落、半減）
- run_1–5 **ペア内**（emb vs matched）は N=20,100 で公平
- しかし **run_0 との before/after 比較は不公平**（N・言語構成が違う）
- F1 低下（0.440→0.349）が mixture のせいかデータ縮小のせいか切り分け不可

### 実装メモ（AI が触った箇所）
- `typology_weighted_sampling()` 追加されたが **`run_condition` から未呼び出し**の可能性（起動ログより）
- run_2–5 の sweep は実質無効（全メトリクス・training_budget 同一）

### Pod 障害
- LaTeX: `apt-get install chktex texlive-latex-base texlive-latex-extra texlive-science`
- writeup 中: OpenAI TPM 30000 < 要求 32271、S2 API 429（長時間バックオフ）
- → **実験完了後 kill、Mac で PDF**

### Mac 成果物
- `results/asian_ner/20260714_220439_typology_weighted_mixture/typology_weighted_mixture.pdf`（4ページ、ネガティブ結果を正直に記載）
- プロット: `low_resource_macro_f1_by_condition.png`, `mongolian_f1_by_condition.png`
- `latex/template.tex` を Mac で修正（図ファイル名・結果表・abstract）

---

## 7. 確立した知見（ストーリー）

### エレベーターピッチ
> 人間が公平な multilingual NER 転移プロトコルを設計 → matched random がクラスタリングを大勝。AI Scientist が cohesion と typology mixture の2案を自律実験 → いずれも matched 未達。LLM は失敗教訓から11案生成。現時点の結論: **固定予算下では言語多様性が typology-guided 同質クラスタに勝つ**。

### 3層 screening ナラティブ
1. **Human:** fair floor 設計 + run_0
2. **AI 実行:** head_final_cohesion + typology_weighted_mixture（各5 run、PDF あり）
3. **LLM 生成:** 11 idea + novelty check

### 次の AI run 候補（未実施）
| 優先 | Idea | 理由 |
|------|------|------|
| 1 | `script_aware_mn_ru_penalty` | 変更範囲狭い、言語学ストーリー、未試 |
| 2 | `low_resource_reweight` | F=9、主指標直結、未試 |
| 避ける | cohesion/mixture 再試行、meta-learning, LaBSE, multitask | 実施済み or スコープ外 |

---

## 8. Git / ブランチ

| ブランチ | 主な内容 | 最新コミット |
|----------|----------|--------------|
| `main` | 人間 floor run_0 | `85cec82` |
| `ai-head-final-cohesion-results` | run #1 tar + ideas 準備 | `21ed444` |
| `ai-typology-weighted-mixture-results` | ideas.json, `--idea-names`, run #2 tar | `40aef99` |

### 未コミット（Mac、2026-07-15 時点の可能性）
- `templates/asian_ner/prompt.json` — AI RUN FAILURE 追記
- `templates/asian_ner/experiment.py` — `--skip_conditions` default `all_mixed`
- `ai_scientist/generate_ideas.py` — `papers is None` bugfix
- `results/` 配下 — `.gitignore` 対象（PDF はローカルのみ）

---

## 9. インフラ・運用メモ

### Pod（RunPod）
- Python 3.11 `.venv`（conda なし）
- `setuptools==69.5.1`（aider 用）
- `openai` 2.x 必須（0.28 だと `RateLimitError` import エラー）
- Git push: **PAT を Password に入力**（パスワード認証不可）
- `ssh.runpod.io` 経由 scp は失敗 → **git push/pull で転送**
- `git checkout -- templates/asian_ner/ideas.json` してから pull（Pod ローカル汚れ注意）

### Mac
- `conda activate ai_scientist`
- LaTeX: MacTeX あり → PDF 生成可能
- `plot.py` は sandbox だと segfault → `all` 権限で実行

### 典型起動コマンド（Pod）
```bash
cd /workspace/AI-Scientist
source .venv/bin/activate
export OPENAI_API_KEY="..."
export S2_API_KEY="..."   # novelty check 時

nohup python launch_scientist.py \
  --skip-idea-generation \
  --skip-novelty-check \
  --model gpt-4o-2024-05-13 \
  --experiment asian_ner \
  --idea-names <IDEA_NAME> \
  > <idea_name>.log 2>&1 &
```

### 1 run あたり所要時間
- 約 **1時間/run** × 最大5 run ≈ 5–6時間 + writeup

---

## 10. コード上の重要ポイント

### EVOLVE-BLOCK（AI が編集可能）
- `cluster_linguistic()`, `cluster_embedding()`
- `_hybrid_lang_distance_matrix()`, `_select_k_by_silhouette()`
- `post_cluster_train_langs()`, `augment_train_raw()`
- `HEAD_FINAL` / `HEAD_INITIAL`, typology penalties, budget logic

### 触ってはいけない
- 6条件の名前、言語プール、`mongolian_f1` キー名、seqeval 評価

### 定数
```python
HEAD_FINAL = {"ja", "ko", "mn"}
HEAD_INITIAL = {"ru"}
```

### 結果 JSON 構造
- `final_info.json`: 条件別 `means`/`stderrs` のみ（集約）
- `detailed_results.json`: `n_train_samples`, `train_langs`, `training_budget`, `metadata` — **サンプル数分析はこちら**

### `launch_scientist.py` 追加機能（`fb0ccff`）
```python
--idea-names typology_weighted_mixture   # カンマ区切りで複数可
```

---

## 11. prompt.json への推奨追記（未実施なら）

### AI RUN FAILURE — typology_weighted_mixture
- run_1–5: emb 0.349 vs matched 0.589（run_0 の 0.440/0.615 より悪化）
- run_1–5 全同一（探索無効）
- embedding が **N=20,100・{mn,ko}** に縮小（floor は N≈40,100・{ja,ko,mn}）
- **禁止:** embedding 条件で N や train_langs を unintentionally 変更
- **要求:** mixture は `augment_train_raw()` で 29言語プールから重み付け、総 N は matched と揃える

---

## 12. 用語集

| 用語 | 意味 |
|------|------|
| **cohesion** | typology 的に同系統言語を同クラスタに固めること |
| **matched random** | 同じ N を 29言語からランダムサンプル（多様性の強さ） |
| **IFN** | Interestingness / Feasibility / Novelty（LLM 自己評価、自動選択には未使用） |
| **floor** | 人間設計の cap=0 fair baseline（run_0） |
| **プロトコル逸脱** | AI が floor と異なる N・言語構成で走り、run_0 との直接比較が不公平になること |

---

## 13. Claude への最初の指示例

```
templates/asian_ner/CLAUDE_HANDOFF.md を読んでコンテキストを復元してください。
次のタスク: [例: prompt.json に run #2 失敗を追記 / script_aware の Pod 起動手順 / screening PDF 用1ページサマリー]
リポジトリは sakuraimai/AI-Scientist、ブランチ ai-typology-diversity-expansion-results（2026-08-11 時点の最新作業）。
```

---

## 14. 主要ファイルパス一覧

```
templates/asian_ner/
  experiment.py          # メイン実験
  prompt.json            # AI プロンプト・floor 結果・失敗教訓
  ideas.json             # 11案
  seed_ideas.json        # 人間 seed 6案
  plot.py
  latex/template.tex
  run_0/                 # 人間 floor
  CLAUDE_HANDOFF.md      # 本ファイル

results/asian_ner/
  20260712_123632_head_final_cohesion/       # AI run #1
  20260714_220439_typology_weighted_mixture/ # AI run #2
    typology_weighted_mixture.pdf
    run_0..run_5/final_info.json
    run_1..run_5/detailed_results.json

launch_scientist.py      # --idea-names 追加済み
ai_scientist/generate_ideas.py
```

---

## 15. AI Run #3 — `typology_guided_diversity_expansion`（2026-08-11、進行中）

### 背景・出発点

Run #1・#2 の失敗を踏まえたディスカッション（Claude とのセッション）から、次の再定義に至った:

> typology（言語学的知見）は「似た言語に絞り込む」ためではなく、「同一予算の中で、どの言語を多様性として加えるべきかを設計する」ために使うべきである。

これを検証するアイデアとして `typology_guided_diversity_expansion` を新規追加し、ブランチ `ai-typology-diversity-expansion-results` で作業中。

### 設計（最終版）

- `cluster_linguistic()`/`cluster_embedding()` は**変更しない**。`BASE_CLUSTER = ["ja","ko","mn"]`（floor と同じ）を定数としてハードコード
- `embedding_clustering`/`linguistic_clustering` は、`augment_train_raw()` で **入れ替え（replacement）** により多様性を導入する: 入れ替え比率 `replacement_ratio`（R）に応じて、BASE_CLUSTER の文をランダムに `n_replace` 文だけ外し、他言語（29言語プールから BASE_CLUSTER を除いた 26 言語）から同数を持ち込む。**N は入れ替えなので常に一定（40,100）**
- 持ち込む言語のランキング基準は条件ごとに異なる（embeddings vs linguistics の対比を保つため）:
  - `linguistic_clustering`: 純粋な typology 距離（HEAD_FINAL/HEAD_INITIAL グループ、embedding 不使用）
  - `embedding_clustering`: 純粋な埋め込み距離（typology ペナルティなし）
- R は 5 run で 0.10, 0.25, 0.50, 0.75, 0.90 程度にスイープ（単調増加・重複禁止）
- `matched_random_embedding`/`matched_random_linguistic`/`all_mixed`/`per_language` には一切影響しない（`augment_train_raw()` は `condition_name` で embedding/linguistic 条件のみに限定）

### 重大な発見（レポート A1〜A4 に反映済み）

1. **`cluster_embedding()` は非決定的。** 同じ `HYBRID_TYPOLOGY_PENALTY=1.0` でも、実行のたびに異なるクラスタ（{ja,ko,mn} になったり {mn,ko} に縮小したり）を返す。埋め込み計算（`_sentence_cls_embeddings()`）に乱数が残っているため。→ **クラスタは再計算せず固定値で与える**必要がある、という設計上の教訓
2. **`prompt.json` は実行フェーズでは一切読まれない。** `perform_experiments()`（実際にコードを書いて走らせる Aider ベースの関数）は `idea["Title"]`/`idea["Experiment"]` のみを受け取り、`prompt.json` は `generate_ideas()`/`check_idea_novelty()` でのみ使われる。`--skip-idea-generation --skip-novelty-check` で起動する限り、`prompt.json` への追記は実行エージェントに一切届かない。**今後、AI への指示は `ideas.json`/`seed_ideas.json` の `Experiment` フィールドに書くこと**（`prompt.json` はあくまで記録・ドキュメントとして書く）

### 試行錯誤の経緯（デバッグの記録）

| 試行 | 内容 | 結果 |
|---|---|---|
| v1（自動生成、抽象的な指示） | 「クラスタリング関数は変えるな、`post_cluster_train_langs()` で言語を typology 距離順に追加せよ」という抽象的な指示 | `embedding_clustering` が非決定性で {mn,ko} N=20,100 に縮小（Run #2 と同じ症状） |
| v2（BASE_CLUSTER ハードコード + 具体的な Dataset 再構築手順を指示） | `pool_indices` の範囲外参照（`IndexError`）→ AI が自己修正するも `raw_ds[i] = ...` という `Dataset` の直接代入（サポート外）で `TypeError`。`MAX_ITERS=4` 到達で `Success: False` | 失敗（自動リトライで解決せず） |
| v3（さらに具体的な仕様を `ideas.json` に明記） | `load_raw_ner_dataset()` で別途候補プールをロードし `concatenate_datasets`/`select()` で再構築せよ、と指示 | `augment_train_raw()` を**全条件**に適用してしまい、`all_mixed` 等で `other_langs` が空になり `ValueError`。再び `Success: False` |
| **人間が直接実装**（コミット `7867158`） | `experiment.py` に `BASE_CLUSTER` 定数、`condition_name` でガードした `augment_train_raw()`、`device` の受け渡しを直接実装。AI へのタスクは「`--replacement_ratio` の default 値を run ごとに変えて実行するだけ」に縮小 | 検証中（2026-08-11 実行中） |

**この経緯自体が、AI Scientist の自律性の限界を示す重要な観察**（レポート §6.2 に「第五の観察点」として追記予定）。特に、v2→v3 の自己修正が「クラッシュは直すが実験の意図を壊す」というパターンを繰り返した点、`MAX_ITERS=4` という小さな予算内では収束しなかった点、最終的に疑似コードレベルの人間の仕様指定が必要だった点、を正確に記録すること。

### 最終結果（2026-08-11、確定・完了）

`run_1`（R=0.75）、`run_2`（R=0.25）、`run_3`（R=0.50）が完走。人間による実装（クラッシュは解消）にもかかわらず、**3 つの R すべてで結果が完全に一致**した:

| Run | R（replacement_ratio） | `embedding_clustering` low_res_f1 | `linguistic_clustering` low_res_f1 | `matched_random_embedding` low_res_f1 | `matched_random_linguistic` low_res_f1 |
|---|---|---|---|---|---|
| run_1 | 0.75 | 0.42590962149604966 | 0.4215014099843493 | 0.6248473239808378 | 0.6135156715831162 |
| run_2 | 0.25 | 0.42590962149604966（同一） | 0.4215014099843493（同一） | 0.6248473239808378（同一） | 0.6135156715831162（同一） |
| run_3 | 0.50 | 0.42590962149604966（同一） | 0.4215014099843493（同一） | 0.6248473239808378（同一） | 0.6135156715831162（同一） |

学習中の loss の値（ステップ 1〜3）も 3 run で完全一致することを確認済み（`typology_guided_diversity_expansion_v3.log` の `[linguistic_clustering] train_budget` 直後の行、行番号 380/12392/24321 付近）。**これは効果が小さいのではなく、3 run で使われた学習データが実質的に同一だったことを意味する。**

**コード上の結線（`RunConfig` フィールド、`build_config()`、argparse のデフォルト値、`augment_train_raw()` 内の参照）はすべて確認済みで正しい。** それでも効果が出ない残りの疑い: `tokenize_ner_dataset()` または HuggingFace `datasets` の `.map()` が、`raw_ds.select()` で作った部分集合を、内容ではなく「指紋」だけで判定し、古いキャッシュを再利用してしまっている可能性（HF datasets の既知の落とし穴の一種）。**時間の制約により未確認・未解決のまま終了とした。**

### 結論の位置づけ

Run #3 は Run #1・#2 とは異なる種類の結果である:
- Run #1・#2: 仮説を検証し、支持されなかった（ネガティブな結果）
- **Run #3: 3 回の実装修正（AI の自動修正 2 回＋人間による直接実装 1 回）を経てクラッシュは解消したが、パラメータを変えても結果が一切変化せず、仮説を検証すること自体ができなかった（inconclusive）**

結果データは取得済み（コミット `e15543b`、`typology_guided_diversity_expansion_results.tar.gz`）。Pod は停止済み。**これ以上の追加実行は行わない、と判断して終了した。**

### 次にやること（レポート側に引き継ぎ）

1. レポート（`Sakana/LLM Agents/report_part5_draft.md`）に §5.4（Run #3 の結果、inconclusive として記載）と §6.2（デバッグ経緯全体を「第五の観察点」として追記）を執筆する
2. 詳細・執筆方針はレポート側の引き継ぎ文書（`HANDOFF_report_part5.md`、Sakana Career/LLM Agents フォルダ）を参照。**そちらが今後の作業の主体となる**

---

*End of handoff memo. Run #3 完了・終了（2026-08-11）。*
