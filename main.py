import json
import os
import re
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

# --- Constants and Setup ---
BINS = [0, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    ".model",
]
TOKENIZER_CACHE_DIR = "tokenizer_cache"
os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)


# --- Caching and Tokenizer Loading ---


def get_cached_tokenizers():
    """Returns a list of directories in the tokenizer cache."""
    if not os.path.exists(TOKENIZER_CACHE_DIR):
        return []
    return [
        d
        for d in os.listdir(TOKENIZER_CACHE_DIR)
        if os.path.isdir(os.path.join(TOKENIZER_CACHE_DIR, d))
    ]


def sanitize_model_path(path):
    """Sanitizes a model name to be a valid directory name."""
    return re.sub(r'[\\/*?:"<>|]', "_", path)


def load_tokenizer(model_name_or_path, model_source, hf_token, progress=gr.Progress()):
    """Loads tokenizer, using a persistent cache to avoid re-downloading."""
    if os.path.isdir(model_name_or_path):  # Prioritize local directory input
        return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    sanitized_name = sanitize_model_path(model_name_or_path)
    cache_path = os.path.join(TOKENIZER_CACHE_DIR, sanitized_name)

    if os.path.exists(cache_path) and os.listdir(cache_path):
        progress(1, desc=f"Loading tokenizer from cache: {sanitized_name}")
        return AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)

    progress(0, desc=f"Downloading tokenizer for {model_name_or_path}")
    os.makedirs(cache_path, exist_ok=True)

    try:
        if model_source == "ModelScope":
            from modelscope.hub.snapshot_download import snapshot_download

            snapshot_download(
                model_name_or_path, local_dir=cache_path, allow_patterns=TOKENIZER_FILES
            )

        elif model_source == "HuggingFace":
            repo_files = list_repo_files(model_name_or_path, token=hf_token)
            tokenizer_repo_files = [
                f
                for f in repo_files
                if Path(f).name in TOKENIZER_FILES or Path(f).name.endswith(".model")
            ]
            if not tokenizer_repo_files:
                raise ValueError("No tokenizer files found in the repository.")

            for file in progress.tqdm(tokenizer_repo_files, desc="Downloading files"):
                hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=file,
                    local_dir=cache_path,
                    token=hf_token,
                )

        return AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)

    except Exception as e:
        raise gr.Error(
            f"Failed to load tokenizer for '{model_name_or_path}'. Error: {e}"
        )


# --- Data Processing and Normalization ---


def normalize_to_messages(item: dict) -> list | None:
    """Detects dataset format (ShareGPT, Alpaca) and converts to a standard 'messages' list."""
    if "conversations" in item and isinstance(item.get("conversations"), list):
        return [
            {
                "role": conv.get("from", "user")
                .replace("human", "user")
                .replace("gpt", "assistant"),
                "content": conv.get("value", ""),
            }
            for conv in item["conversations"]
        ]
    elif "messages" in item and isinstance(item.get("messages"), list):
        return item["messages"]
    elif "instruction" in item and "output" in item:
        user_content = item.get("instruction", "")
        if item.get("input"):
            user_content += f"\n{item['input']}"
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item.get("output", "")},
        ]
    return None


def apply_chat_template(tokenizer, item, template):
    """Applies a chat template to a single data item after normalizing its format."""
    try:
        if template == "raw":
            if "text" in item and isinstance(item["text"], str):
                return item["text"]
            if "content" in item and isinstance(item["content"], str):
                return item["content"]
            for value in item.values():
                if isinstance(value, str):
                    return value
            return str(item)

        messages = normalize_to_messages(item)
        if messages is None:
            raise ValueError(
                "Unsupported data format. Expected ShareGPT-style or Alpaca-style."
            )

        template_arg = template if template != "auto" else None
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=template_arg,
        )

    except Exception as e:
        return f"TEMPLATE_ERROR: {e}"


# --- Main Analysis Function ---


def analyze_dataset(
    file_obj,
    model_name_or_path,
    model_source,
    chat_template,
    hf_token,
    adaptive_bins,
    progress=gr.Progress(),
):
    if file_obj is None:
        raise gr.Error("Please upload a dataset file.")
    if not model_name_or_path:
        raise gr.Error("Please specify a model name or path.")

    tokenizer = load_tokenizer(model_name_or_path, model_source, hf_token, progress)

    try:
        tmp_file_path = file_obj
        file_extension = os.path.splitext(tmp_file_path)[1].lower()
        loader = {
            ".jsonl": "json",
            ".json": "json",
            ".csv": "csv",
            ".parquet": "parquet",
        }.get(file_extension)
        if not loader:
            raise gr.Error(f"Unsupported file type: {file_extension}.")
        dataset = load_dataset(loader, data_files=tmp_file_path)["train"]
    except Exception as e:
        raise gr.Error(f"Failed to load dataset. Error: {e}")

    token_counts, total_tokens, processing_errors = [], 0, []
    for i, item in enumerate(progress.tqdm(dataset, desc="Tokenizing Dataset...")):
        formatted_text = apply_chat_template(tokenizer, item, chat_template)
        if isinstance(formatted_text, str) and formatted_text.startswith(
            "TEMPLATE_ERROR:"
        ):
            processing_errors.append(
                f"Row {i}: {formatted_text.replace('TEMPLATE_ERROR: ', '')}"
            )
            continue
        tokens = tokenizer.encode(formatted_text)
        token_counts.append(len(tokens))

    if not token_counts:
        raise gr.Error(
            "Could not process any items. Check dataset format and template compatibility."
        )
    total_tokens = sum(token_counts)

    # --- Plotting Logic ---
    fig, ax = plt.subplots(figsize=(12, 7))

    if adaptive_bins:
        bin_width = 256
        max_val = max(token_counts)
        bins = np.arange(0, max_val + bin_width, bin_width)
        counts, edges = np.histogram(token_counts, bins=bins)

        non_zero_indices = np.where(counts > 0)[0]
        if len(non_zero_indices) > 0:
            start_index = max(0, non_zero_indices[0] - 1)
            end_index = min(len(counts), non_zero_indices[-1] + 2)
            counts = counts[start_index:end_index]
            edges = edges[start_index : end_index + 1]

        labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(counts))]
        values = counts
    else:
        bin_counts = {
            label: 0
            for label in [f"{BINS[i]}-{BINS[i + 1]}" for i in range(len(BINS) - 1)]
            + [f">{BINS[-1]}"]
        }
        for count in token_counts:
            for i in range(len(BINS) - 1):
                if BINS[i] < count <= BINS[i + 1]:
                    bin_counts[f"{BINS[i]}-{BINS[i + 1]}"] += 1
                    break
            else:
                if count > BINS[-1]:
                    bin_counts[f">{BINS[-1]}"] += 1
        labels, values = zip(*bin_counts.items())

    bars = ax.bar(labels, values, color="skyblue")
    ax.bar_label(bars)
    ax.set_title("Token Count Distribution")
    ax.set_xlabel("Token Count Range")
    ax.set_ylabel("Number of Items")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()

    # --- Finalization ---
    temp_dataset_path = tempfile.mktemp(suffix=".jsonl")
    with open(temp_dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    if processing_errors:
        gr.Warning(
            f"Encountered {len(processing_errors)} errors. First 5:\n"
            + "\n".join(processing_errors[:5])
        )

    # Return new Dropdown choices to refresh cached list
    return (
        f"Total Tokens: {total_tokens}",
        fig,
        gr.Slider(1, len(dataset), step=1, label="Preview Dataset Item"),
        gr.JSON(dataset[0]),
        temp_dataset_path,
        gr.Dropdown(choices=get_cached_tokenizers()),
    )


def get_dataset_item(dataset_path, index):
    if not dataset_path or not os.path.exists(dataset_path):
        return "Dataset not loaded yet."
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index - 1:
                return json.loads(line)
    return "Index out of bounds."


# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# Fine-Tuning Dataset Token Analyzer")
    gr.Markdown(
        "Features: Caching, adaptive bins, and auto-detection of ShareGPT/Alpaca formats."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload Dataset (.json, .jsonl, .csv, .parquet)"
            )

            gr.Markdown("### Tokenizer Settings")
            model_source = gr.Radio(
                ["ModelScope", "HuggingFace"],
                label="Tokenizer Source",
                value="ModelScope",
            )

            model_name_input = gr.Dropdown(
                label="Model Name or Local Path",
                info="Select a cached model or type a new one to download.",
                choices=get_cached_tokenizers(),
                allow_custom_value=True,
            )
            hf_token = gr.Textbox(
                label="Hugging Face Token (optional)", type="password"
            )

            gr.Markdown("### Processing Settings")
            chat_template = gr.Dropdown(
                ["auto", "raw", "chatml", "llama2", "zephyr", "qwen", "qwen2"],
                label="Chat Template",
                value="auto",
                info="'auto' uses the tokenizer's default. 'raw' tokenizes text directly.",
            )
            adaptive_bins_checkbox = gr.Checkbox(
                label="Enable Adaptive Bins", value=True
            )

            analyze_btn = gr.Button("Analyze Dataset", variant="primary")

        with gr.Column(scale=2):
            total_tokens_out = gr.Textbox(label="Statistics", interactive=False)
            distribution_plot = gr.Plot(label="Token Distribution")

    gr.Markdown("### Dataset Preview")
    temp_dataset_path_state = gr.State(None)
    preview_slider = gr.Slider(
        label="Preview Dataset Item", minimum=1, maximum=1, step=1, interactive=True
    )
    item_preview = gr.JSON(label="Dataset Item Content")

    # --- Event Handlers ---
    analyze_btn.click(
        fn=analyze_dataset,
        inputs=[
            file_upload,
            model_name_input,
            model_source,
            chat_template,
            hf_token,
            adaptive_bins_checkbox,
        ],
        outputs=[
            total_tokens_out,
            distribution_plot,
            preview_slider,
            item_preview,
            temp_dataset_path_state,
            model_name_input,  # Update dropdown choices on completion
        ],
    )
    preview_slider.release(
        fn=get_dataset_item,
        inputs=[temp_dataset_path_state, preview_slider],
        outputs=[item_preview],
    )

if __name__ == "__main__":
    demo.launch()
