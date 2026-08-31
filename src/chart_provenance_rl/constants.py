from pathlib import Path

SCHEMA_VERSION = "1.0"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_LICENSE = "Apache-2.0"
MAX_EVIDENCE_REFS = 8
DEFAULT_IMAGE_SIZE = (512, 384)
DEFAULT_DATASET_ROOT = Path("data/generated")

SYSTEM_PROMPT = (
    "Answer the chart question using only the visible image. Return one JSON object and no "
    "other text. The object must have exactly this shape: "
    '{"answer":"...","evidence":[{"series":"Revenue","category":"Q3"}]}. '
    "Evidence must name the visible series and category labels that are minimally sufficient "
    f"for the answer. Use at most {MAX_EVIDENCE_REFS} evidence items."
)
