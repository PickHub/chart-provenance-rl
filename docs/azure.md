# Azure ML execution

## Workspace configuration

No subscription, resource-group, workspace identifier, or credential is stored in this
public repository. Set the deployment target in the shell:

```bash
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
export AZURE_RESOURCE_GROUP="<resource-group>"
export AZURE_ML_WORKSPACE="<workspace-name>"
export AZURE_ML_COMPUTE="gpu-t4-dedicated"
export AZURE_LOCATION="spaincentral"

az login
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
az configure --defaults \
  group="$AZURE_RESOURCE_GROUP" \
  workspace="$AZURE_ML_WORKSPACE" \
  location="$AZURE_LOCATION"
```

## Mandatory live preflight

Run this only immediately before an Azure integration or study run:

```bash
uv run python scripts/azure_preflight.py
```

The preflight queries `az ml compute list-sizes` and fails unless
`Standard_NC4as_T4_v3` supports `AmlCompute`. It also verifies the selected cluster.
During implementation review on 2026-08-31, `Standard_NV36ads_A10_v5` exposed only
`MIR`, not `AmlCompute`, and raw-VM A10 quota was 8 of the 36 required cores. The A10
path is unavailable, not ready-but-optional.

## Validate and register assets

The environment uses the immutable base image:

```text
mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04@sha256:2f88524bcaaaeecd457236fe6ab35965a780640449402c57aab14de91f5569b9
```

```bash
uv run --extra azure python scripts/validate_azure_assets.py

az ml environment create \
  --file azure/environment/environment.yml \
  --subscription "$AZURE_SUBSCRIPTION_ID" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE"
```

The T4 cluster should already exist. Recreate it only if preflight confirms the same
SKU:

```bash
az ml compute create \
  --file azure/compute/t4.yml \
  --subscription "$AZURE_SUBSCRIPTION_ID" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE"
```

There is no SSH configuration. The cluster scales to zero.

## Required integration smoke

Before full runs, submit one T4 integration smoke with `profile=smoke`, one SFT step,
one GRPO step, and `validation` evaluation. This exercises the real trainer
constructors, plural-image SFT collator, post-PEFT vision freeze, float16 quantized-model
compute with float32 adapter gradients, processor pixel cap, generation, rewards, and
checkpoint writing. Its metrics are not study results.

```bash
az ml job create \
  --file azure/pipeline.yml \
  --subscription "$AZURE_SUBSCRIPTION_ID" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --workspace-name "$AZURE_ML_WORKSPACE" \
  --set inputs.profile=smoke \
        inputs.seed=17 \
        inputs.sft_steps=1 \
        inputs.grpo_steps=1 \
        inputs.evaluation_split=validation \
  --stream
```

## Submit matched seeds

Use fixed datastore paths so retrying a seed sees its `checkpoint-*` directories.
Immutable run manifests reject seed, reward-arm, data, dependency, or optimizer config
drift before resume.

```bash
for SEED in 17 29 43; do
  az ml job create \
    --file azure/pipeline.yml \
    --subscription "$AZURE_SUBSCRIPTION_ID" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workspace-name "$AZURE_ML_WORKSPACE" \
    --set inputs.seed="$SEED" \
          outputs.dataset.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/seed-$SEED/dataset" \
          outputs.model_snapshot.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/model/89644892e4d85e24eaac8bacfd4f463576704203" \
          outputs.sft_adapter.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/seed-$SEED/sft" \
          outputs.grpo_answer_adapter.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/seed-$SEED/grpo-answer" \
          outputs.grpo_naive_adapter.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/seed-$SEED/grpo-naive" \
          outputs.grpo_minimal_adapter.path="azureml://datastores/workspaceblobstore/paths/chart-provenance-rl/seed-$SEED/grpo-minimal" \
    --stream
done
```

The pipeline evaluates the minimal adapter normally, with a neutral image, and with a
deterministic derangement of held-out images. The latter two are the actual model
leakage probes. The template-majority question-only baseline remains descriptive.

## Collect live throughput evidence

The confirmatory gate does not accept a hand-written hardware label. After all three
seed pipelines finish, collect their parent and child job metadata plus downloaded
run/adapter manifests:

```bash
uv run chart-provenance collect-azure-throughput \
  --pipeline-job "<seed-17-parent-job>" \
  --pipeline-job "<seed-29-parent-job>" \
  --pipeline-job "<seed-43-parent-job>" \
  --output results/t4-throughput.json
```

The collector uses the Azure environment variables defined above. It queries each
parent pipeline and `az ml job list --parent-job-name`, verifies completed SFT,
naive, and minimal child jobs on the configured AmlCompute cluster, checks the live
cluster SKU, derives wall time from Azure timestamps, downloads each child artifact,
validates adapter weights, and records source/config/data/run/adapter hashes. Missing,
nonterminal, wrong-seed, wrong-compute, or mismatched evidence aborts collection.
Full training-config hashes retain each run's seed. A second hash removes only `seed`
and `data_seed` when checking that all other configuration is identical across seeds.

The resulting artifact follows
[`schemas/t4-throughput.schema.json`](../schemas/t4-throughput.schema.json). It stores
job IDs, status, SKU, timestamps, and hashes, but no account IDs or user creation
metadata.

## Confirmatory analysis

Concatenate each method's `question_metrics.jsonl` and naive/minimal
`rollout_diagnostics.jsonl` over seeds, preserving their provenance fields. Then run:

```bash
uv run chart-provenance confirm \
  --sft results/sft.jsonl \
  --naive results/grpo-naive.jsonl \
  --minimal results/grpo-minimal.jsonl \
  --question-only results/grpo-minimal-question-only.jsonl \
  --shuffled results/grpo-minimal-shuffled.jsonl \
  --dataset-audit results/dataset-audit.json \
  --throughput results/t4-throughput.json \
  --rollout-diagnostics results/naive-rollouts.jsonl results/minimal-rollouts.jsonl \
  --output results/confirmatory-report.json
```

The report is fail-closed. Missing artifacts produce `incomplete`; malformed,
unmatched, wrong-seed, wrong-split, or wrong-method artifacts produce `fail`. Rollout
rows are summarized as non-independent diagnostics and never mixed into paired
question-level inference.
