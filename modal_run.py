import sys
import modal

# ── 1. Define the cloud environment (the "image") ────────────────────────────
# linux + cuda is the standard ml runtime. models, scripts, training engines typically assume linux + nvidia gpu + cuda
# nano=vLLM is a CUDA-native software (triton kernels + cuda graphs + gpu torch). CUDA only lives properly on Linux + NVIDIA, and because this mac has neither, we use this container instead.
#
# nano-vllm depends on flash-attn, a CUDA extension that normally COMPILES from
# source (needs nvcc, lots of RAM, ~20 min). We skip that by installing a PREBUILT
# wheel. The wheel filename pins everything that must agree:
#   cu12 -> CUDA 12  |  torch2.6 -> PyTorch 2.6  |  cp312 -> Python 3.12
# Python 3.12 also satisfies nano-vllm's requirement of >=3.10,<3.13.
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",          # pulls in a matching triton automatically
        "transformers>=4.51.0",
        "huggingface_hub",
        "xxhash",
        "einops",                # flash-attn runtime helper
        flash_attn_wheel,        # prebuilt -> no compilation
    )
    # Mount YOUR local source on top so your edits override the deps' copy.
    # copy=False => mounted at run time, so editing locally + re-running picks up
    # changes WITHOUT rebuilding the image.
    .add_local_dir(
        "/Users/robert/nano-vllm/nanovllm",
        "/root/nanovllm_local/nanovllm",
        copy=False,
    )
)

# ── 2. A persistent disk for model weights ───────────────────────────────────
# Without this, you'd re-download Qwen3-0.6B on every run. The Volume caches it
# across runs so only the first run pays the download cost.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("nano-vllm-learning", image=image)


# ── 3. The function that runs ON the GPU ─────────────────────────────────────
@app.function(
    gpu="L4",                                   # Ada GPU; flash-attn 2 needs Ampere+ (T4 won't work)
    timeout=600,                                # kill if it hangs past 10 min
    volumes={"/cache": hf_cache},               # mount the Volume at /cache
)
def generate(prompts: list[str]):
    import os
    # Send HuggingFace's cache to the Volume so weights persist across runs.
    os.environ["HF_HOME"] = "/cache/huggingface"

    # Make YOUR mounted source win over anything in site-packages.
    sys.path.insert(0, "/root/nanovllm_local")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    hf_cache.commit()                           # persist the download to the Volume

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # enforce_eager=True skips CUDA-graph capture -> faster startup while learning.
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling = SamplingParams(temperature=0.6, max_tokens=256)

    # Qwen3 is a CHAT model: raw strings -> garbage. Wrap each in its chat template.
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    outputs = llm.generate(chat_prompts, sampling)
    return [o["text"] for o in outputs]         # each output is a dict with 'text'


# ── 4. The local entrypoint — runs on YOUR Mac, calls the GPU function ───────
@app.local_entrypoint()
def main():
    prompts = ["introduce yourself", "list all prime numbers within 100"]
    results = generate.remote(prompts)          # .remote() = run in the cloud
    for p, r in zip(prompts, results):
        print(f"\nPROMPT: {p}\nOUTPUT: {r}")
