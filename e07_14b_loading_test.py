import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL = "/data/anouar/EMRLL/E04/model/DeepSeek-R1-Distill-Qwen-14B"

print("=" * 80)
print("E07 DEEPSEEK-R1-DISTILL-QWEN-14B LOADING TEST")
print("=" * 80)

assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2,
    ),
    "GB",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer: OK")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading 14B model with 4-bit quantization + CPU offload...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={
        0: "20GiB",
        "cpu": "80GiB",
    },
    offload_folder="/data/anouar/EMRLL/E04/offload_e07",
    local_files_only=True,
)

print("Model:", model.__class__.__name__)
print(
    "Embedding device:",
    model.get_input_embeddings().weight.device,
)

print("\nDevice map:")
for device, count in model.hf_device_map.items():
    print(f"  {device}: {count}")

prompt = (
    "Solve this problem and give only the final numeric answer:\n"
    "What is 12 + 25?"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

print("\nGenerating test answer...")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(
    output[0],
    skip_special_tokens=True,
)

print("\nGenerated:")
print(generated)

print("\n" + "=" * 80)
print("E07 14B LOADING TEST: PASSED")
print("=" * 80)
