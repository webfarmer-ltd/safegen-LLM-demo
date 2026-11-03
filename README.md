# Gemma 4 E2B × SafeGen型 SFT → GRPO PoC のデモコード

> **注意**: これは SafeGen-LLM の学習原理（SFT → GRPO、5段階の安全報酬）を試す **Blocksworld向けの独自ミニ実装**。論文が用いる PDDL3/OPTIC/VAL データセットの完全再現ではない。GPUでのSFT/GRPOトレーニングはこの配布環境では未実行。CPU-only の報酬関数と生成データのテストは実行済み。

## 追加ファイルと役割

```text
ollama-gemma4/
├── Dockerfile.safegen                # 独立した Python3.11/CUDA12.4 学習環境
├── README.md                         # 本ファイル
├── src/
│   ├── safegen_blocks.py             # 記号的な状態遷移・安全検証・報酬
│   ├── safegen_reward.py             # GRPO向けReward callback
│   ├── safegen_val_reward.py         # VAL実装用の接続口（任意）
│   ├── make_safegen_dataset.py       # 教師データ・GRPOデータ生成
│   ├── train_sft_safegen.py          # Gemma4 E2B QLoRA SFT
│   ├── train_grpo_safegen.py         # SFTアダプタから GRPO
│   └── evaluate_safegen.py           # SFT/GRPO の成功率を計測
├── tests/test_reward.py              # CPU側ユニットテスト
└── data/safegen_*.jsonl              # 学習/テスト用ダミーデータ
```

## 1. コンテナのビルド

RTX 3090 24GBを想定。**Python 3.11、PyTorch 2.6/cu124のイメージ**を作る。PyTorchのCUDA対応wheelにランタイムライブラリが含まれるため、ベースイメージは `python:3.11-slim-bookworm` を使用する。

```bash
docker build -f Dockerfile.safegen -t safegen:latest .
```

Gemma 4のモデル読み込みAPIは更新が多い。初回ビルド成功後、`pip freeze`を保存して実際に動作したバージョンへ固定すること。ホスト側にNVIDIA Container ToolkitとCUDA12.4をサポートするドライバーが必要。既存のOllamaコンテナには一切影響しない。

## 2. コンテナ起動
実行例。私の場合、 `/media/ohmasa/hdd2/ohmasa/hf_cache:/workspace/.cache/huggingface` にgemma4-E2Bのパラメータがあるので、そこをパイプする。

```bash
docker run --gpus all -it --rm \
  --name gemma4-safegen \
  -v "$(pwd)":/workspace \
  -v /media/ohmasa/hdd2/ohmasa/hf_cache:/workspace/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  safegen:latest
```

Hugging Faceキャッシュが完備されている前提。初回ダウンロードが必要なら `HF_HUB_OFFLINE=1` を外す。CUDAとライブラリの確認：

```bash
python3 -c "import torch, transformers, trl, peft; print(torch.__version__, transformers.__version__, trl.__version__, peft.__version__, torch.cuda.is_available())"
python3 -m pip freeze > /workspace/outputs/safegen_requirements_frozen.txt  # outputs/作成後
```

## 3. まずCPUの報酬関数をテスト

```bash
cd /workspace
python3 -m unittest discover -s tests -v
```

`format_error`, `safety_violation`, `precondition_violation`, `goal_not_satisfied`, `success` の全カテゴリと、BFSで生成した正解計画の妥当性を検証する。

## 4. データセットの生成

配布ZIPには小さなサンプルが含まれる。生成し直す場合：

```bash
python3 src/make_safegen_dataset.py --out-dir /workspace/data \
  --train-size 80 --test-size 16
```

**SFT用** `data/safegen_train_sft.jsonl` は `prompt` / `completion` で、正解のAction Sequenceを教師として学習する。

```json
{"prompt":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"completion":[{"role":"assistant","content":"(pickup b1)\n(stack b1 b2)"}]}
```

**GRPO用** `data/safegen_train_grpo.jsonl` は `prompt` / `problem_json` で、正解Planは学習時の入力に与えない。

```json
{"prompt":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"problem_json":"{...}"}
```

`problem_json` はモデルへの入力ではなくReward関数へのメタデータである。

## 5. SFT（Stage I）

```bash
cd /workspace
python3 src/train_sft_safegen.py \
  --train-jsonl /workspace/data/safegen_train_sft.jsonl \
  --output /workspace/outputs/gemma4-e2b-safegen-sft
```

`completion_only_loss=True` によって、タスク指示ではなくAssistantのAction Sequenceを学習対象とする。4bitのベースモデルは固定し、LoRAアダプタのみ学習する。元のスクリプトの`modules_to_save=["lm_head", "embed_tokens"]`は新しいトークンを追加しないので省略している。Gemma 4では言語側に限定したLoRAを推奨する。

## 6. GRPO（Stage II）

```bash
python3 src/train_grpo_safegen.py \
  --sft-adapter /workspace/outputs/gemma4-e2b-safegen-sft \
  --train-jsonl /workspace/data/safegen_train_grpo.jsonl \
  --output /workspace/outputs/gemma4-e2b-safegen-grpo \
  --steps 50 --generations 4
```

**SFT済みアダプタを必ず読み込んでからGRPOを実行する。** 新しいLoRAをゼロから追加するのではない。

GRPOは同じ入力から複数のAction Sequenceを確率的に生成する。生成結果を `safegen_reward()` が採点し、TRLがAdvantage・Policy Gradient・KLを計算して更新する。

論文の本設定は候補8、Temperature=0.6、KL係数0.01。こちらは3090の初期PoC向けに候補4、生成長128、学習50 stepsへ縮小している。``scale_rewards=False`` は論文式(5)の平均差分によるAdvantageに合わせている。実装上はPPOクリッピングや長さ正規化の扱いがあるため、論文の全数式の厳密な再現ではない。

## 7. 成功率を比較

```bash
python3 src/evaluate_safegen.py \
  --adapter /workspace/outputs/gemma4-e2b-safegen-sft

python3 src/evaluate_safegen.py \
  --adapter /workspace/outputs/gemma4-e2b-safegen-grpo
```

別のseedのtest問題に対して `success` / `safety_violation` などのカテゴリ別件数を集計する。80件という少ない学習データでの改善は保証されない。学習の有効性を比較するには、複数seedと未学習ベースモデルも測定すること。

## 8. SafeGen論文のPDDL3/VALに近づけるには

このミニ実装の状態遷移シミュレータは自前のものであり、**PDDL3ファイルを直接解釈してはいない**。論文を本格再現するなら、次の変更が必要である。

1. PDDL3ドメイン・問題・制約ファイルを作成する。
2. OPTICなどのPlannerで教師Planを取得し、VALで正解を検証する。
3. `safegen_val_reward.py` の `val_grade()` を使い、VALのstdout/stderrを解析する。VALの実装・出力形式によって解析ルールを合わせる。
4. VALで特定できる最初の違反ステップ `t_v` を抽出し、`L_ref` で正規化する。
5. 最終状態の成立したGoal述語数 `n_sat / n_total` は、VALの出力だけに依存せず、状態遷移評価器を用意して求める。
6. `train_grpo_safegen.py` の `reward_funcs=safegen_reward` をPDDL3/VAL版に差し替える。

`src/safegen_val_reward.py` は**本番再現用の完成品ではない**。この時点ではBlocksWorldの演算子表現だけを受け付け、失敗ステップやGoal達成数はVALの標準出力に依存し、取得できない場合は保守的な報酬を返す。誤って未知のVAL出力を成功扱いしない設計にしてある。

## 数式との対応

- SFT: `completion_only_loss=True` による正解計画トークンのCross Entropy。
- GRPO: `safegen_reward(completions, problem_json, ...)` が1出力につき1報酬を返す。
- Format error `-1.0` / Safety `[-0.9,-0.6]` / Precondition `[-0.6,-0.3]` / Goal `[-0.4,-0.1]` / Success `+1.0`。
- Progress: Constraint violationの初回ステップ/正解長、Goal未達成では目標条件の達成率。
- KL: `beta=0.01`。VRAMが不足する場合は `--beta 0` の実験も可能だが論文と条件が変わる。

## 参考文献

- [SafeGen-LLM (arXiv:2602.24235)](https://arxiv.org/html/2602.24235)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Google Gemma 4 QLoRA guide](https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora)
