"""Fine-tuning package (empty by design).

Add a ``train.py`` that:

1. Loads ``data/finetune/train.jsonl`` (~500 instruction pairs mapping
   free-form experimental descriptions to a validated protocol JSON schema).
2. Uses **Unsloth** to LoRA-fine-tune ``medgemma-4b`` with QLoRA.
3. Saves the adapter to ``$BIOLAB_ARTIFACT_DIR/lora-protocol``.
4. Logs metrics to W&B, MLflow, or TensorBoard.

The adapter is evaluated by re-running the benchmark with
``BIOLAB_LORA_ADAPTER=/artifacts/lora-protocol``. It must improve
*protocol-format validity* by at least 10 percentage points on the
held-out split.

The held-out split is at ``data/finetune/eval.jsonl``  -  do not train on it.
"""
