from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from chart_provenance_rl.constants import (
    DEFAULT_IMAGE_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class ModelStack:
    model: Any
    processor: Any
    peft_config: Any | None


def load_model_stack(
    *,
    model_source: str | Path = MODEL_ID,
    adapter_path: Path | None = None,
    trainable_adapter: bool = True,
    quantized: bool = True,
    local_files_only: bool = False,
) -> ModelStack:
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL training requires an NVIDIA CUDA device")
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        if quantized
        else None
    )
    source = str(model_source)
    source_kwargs: dict[str, Any] = {}
    if source == MODEL_ID:
        source_kwargs["revision"] = MODEL_REVISION
        source_kwargs["local_files_only"] = local_files_only
    else:
        from chart_provenance_rl.training.prepare import validate_model_snapshot

        validate_model_snapshot(Path(source), verify_hashes=True)
        source_kwargs["local_files_only"] = True
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=quantization_config,
        **source_kwargs,
    )
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    freeze_vision_tower(model)
    processor = AutoProcessor.from_pretrained(
        source,
        padding_side="left",
        **source_kwargs,
    )
    processor.image_processor.max_pixels = DEFAULT_IMAGE_SIZE[0] * DEFAULT_IMAGE_SIZE[1]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=trainable_adapter,
        )
        freeze_vision_tower(model)
        return ModelStack(model=model, processor=processor, peft_config=None)
    return ModelStack(model=model, processor=processor, peft_config=lora_config)


def freeze_vision_tower(model: Any) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        visual = base_model.model.visual
    except AttributeError as error:
        raise TypeError("Expected Qwen3-VL model.model.visual vision tower") from error
    for parameter in visual.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in visual.parameters()):
        raise RuntimeError("Vision tower freeze verification failed")


def enforce_trainable_parameter_dtype(model: Any, dtype: Any) -> int:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Model has no trainable adapter parameters")
    for parameter in trainable_parameters:
        parameter.data = parameter.data.to(dtype)
    if any(parameter.dtype != dtype for parameter in trainable_parameters):
        raise RuntimeError(f"Trainable parameter dtype verification failed for {dtype}")
    return sum(parameter.numel() for parameter in trainable_parameters)


def model_stack_smoke(
    *,
    model_source: str | Path = MODEL_ID,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import get_peft_model

    stack = load_model_stack(
        model_source=model_source,
        local_files_only=local_files_only,
    )
    if stack.peft_config is None:
        raise RuntimeError("Model smoke requires a fresh LoRA configuration")
    model = get_peft_model(stack.model, stack.peft_config)
    freeze_vision_tower(model)
    trainable = enforce_trainable_parameter_dtype(model, torch.float16)
    image = Image.new("RGB", DEFAULT_IMAGE_SIZE, "white")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What is visible?"},
            ],
        },
    ]
    inputs = stack.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=1)
    if generated.shape[1] <= inputs["input_ids"].shape[1]:
        raise RuntimeError("Model generation smoke produced no completion token")
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash is not None and commit_hash != MODEL_REVISION:
        raise RuntimeError(f"Loaded model revision {commit_hash}, expected {MODEL_REVISION}")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": "4bit-nf4-double-quant",
        "compute_dtype": "float16",
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 8,
        "vision_frozen": True,
        "trainable_parameter_count": trainable,
        "generated_token_count": 1,
    }
