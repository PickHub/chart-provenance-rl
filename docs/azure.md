# Azure ML execution

## Fixed workspace

```text
subscription:   eb5c4bc0-bf07-4a49-b683-37be7d6f0e94
resource group: rg-rag-support-scorer
workspace:      aml-rag-support-scorer
region:         Spain Central
```

No secret is stored in this repository. Authenticate interactively or through the
calling environment's managed identity.

```bash
az login
az account set --subscription eb5c4bc0-bf07-4a49-b683-37be7d6f0e94
az configure --defaults \
  group=rg-rag-support-scorer \
  workspace=aml-rag-support-scorer \
  location=spaincentral
```

## Mandatory live preflight

```bash
az ml compute list-sizes \
  --location spaincentral \
  --subscription eb5c4bc0-bf07-4a49-b683-37be7d6f0e94 \
  --resource-group rg-rag-support-scorer \
  --workspace-name aml-rag-support-scorer \
  --query "[?name=='Standard_NV36ads_A10_v5' || name=='Standard_NC4as_T4_v3'].{name:name,supportedComputeTypes:supportedComputeTypes}" \
  --output json

az ml compute show \
  --name gpu-t4-dedicated \
  --subscription eb5c4bc0-bf07-4a49-b683-37be7d6f0e94 \
  --resource-group rg-rag-support-scorer \
  --workspace-name aml-rag-support-scorer \
  --output table

uv run python scripts/azure_preflight.py
```

Observed on 2026-08-31:

- `Standard_NC4as_T4_v3` supports `AmlCompute`; cluster `gpu-t4-dedicated`
  has min 0, max 1, and 300-second idle timeout.
- `Standard_NV36ads_A10_v5` reports only `MIR`, so Azure ML cluster creation
  fails with `InvalidPropertyValue`.
- raw-VM quota request `db2ba018-2a40-48ab-8f4c-8b11a55014e0` was
  `PartiallyFulfilled` at 8 A10-family cores; the VM requires 36.

The A10 path is unavailable, not ready-but-optional. Do not create an A10 cluster asset
until both supported compute type and quota are revalidated.

## Validate and register assets

The environment uses the immutable base image:

```text
mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04@sha256:2f88524bcaaaeecd457236fe6ab35965a780640449402c57aab14de91f5569b9
```

```bash
uv run --extra azure python scripts/validate_azure_assets.py

az ml environment create \
  --file azure/environment/environment.yml \
  --subscription eb5c4bc0-bf07-4a49-b683-37be7d6f0e94 \
  --resource-group rg-rag-support-scorer \
  --workspace-name aml-rag-support-scorer
```

The T4 cluster already exists. Recreate it only if preflight confirms the same SKU:

```bash
az ml compute create \
  --file azure/compute/t4.yml \
  --subscription eb5c4bc0-bf07-4a49-b683-37be7d6f0e94 \
  --resource-group rg-rag-support-scorer \
  --workspace-name aml-rag-support-scorer
```

There is no SSH configuration. The cluster scales to zero.

## Submit matched seeds

Use fixed datastore paths so retrying the same seed sees `checkpoint-*` directories.
The code resolves the highest checkpoint with `--resume auto`.

Before full runs, submit one T4 integration smoke by setting `inputs.profile=smoke`,
`inputs.sft_steps=1`, `inputs.grpo_steps=1`, and
`inputs.evaluation_split=validation`. This is the required trainer-construction,
adapter-dtype, generation, reward, and checkpoint check. Do not treat its metrics as a
study result.

```bash
SUB=eb5c4bc0-bf07-4a49-b683-37be7d6f0e94
RG=rg-rag-support-scorer
WS=aml-rag-support-scorer

for SEED in 17 29 43; do
  az ml job create \
    --file azure/pipeline.yml \
    --subscription "$SUB" \
    --resource-group "$RG" \
    --workspace-name "$WS" \
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

The environment, component versions, model revision, and source snapshot are
immutable for a submitted job. MLflow records trainer scalars. Each GRPO adapter also
contains `rollout_diagnostics.jsonl` with selected reward and hidden precision/joint
metrics. Those rows are diagnostics, not statistical units.

Because the cluster max is one, GPU nodes queue. This is intentional cost control.

## Analyze

Download each job's `question_metrics.jsonl`, concatenate the same method across seeds,
then run:

```bash
uv run chart-provenance analyze \
  --treatment results/grpo-minimal.question-metrics.jsonl \
  --control results/grpo-naive.question-metrics.jsonl \
  --metric joint_exact \
  --output results/primary-analysis.json
```

`azure/components/analyze.yml` provides the same analysis as an Azure command
component when the two concatenated files are registered as immutable inputs. Analysis
fails unless both files contain the same `(record_id, seed)` pairs and split, and each
file contains exactly one distinct method label.
