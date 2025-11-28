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
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "merges.txt", "special_tokens_map.json", ".model"]
TOKENIZER_CACHE_DIR = "tokenizer_cache"
os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)

# --- Caching and Tokenizer Loading ---

def get_cached_tokenizers():
    if not os.path.exists(TOKENIZER_CACHE_DIR): return []
    return [d for d in os.listdir(TOKENIZER_CACHE_DIR) if os.path.isdir(os.path.join(TOKENIZER_CACHE_DIR, d))]

def sanitize_model_path(path):
    return re.sub(r'[\\/*?:"<>|]', '_', path)

def load_tokenizer(model_name_or_path, model_source, hf_token, progress=gr.Progress()):
    if os.path.isdir(model_name_or_path):
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
            snapshot_download(model_name_or_path, local_dir=cache_path, allow_patterns=TOKENIZER_FILES)
        elif model_source == "HuggingFace":
            repo_files = list_repo_files(model_name_or_path, token=hf_token)
            tokenizer_repo_files = [f for f in repo_files if Path(f).name in TOKENIZER_FILES or Path(f).name.endswith('.model')]
            if not tokenizer_repo_files: raise ValueError("No tokenizer files found in repository.")
            for file in progress.tqdm(tokenizer_repo_files, desc="Downloading files"):
                hf_hub_download(repo_id=model_name_or_path, filename=file, local_dir=cache_path, token=hf_token)
        return AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
    except Exception as e:
        raise gr.Error(f"Failed to load tokenizer for '{model_name_or_path}'. Error: {e}")

# --- Data Processing and Normalization ---

def normalize_to_messages(item: dict) -> list | None:
    if 'conversations' in item and isinstance(item.get('conversations'), list):
        return [{'role': conv.get('from', 'user').replace('human', 'user').replace('gpt', 'assistant'), 'content': conv.get('value', '')} for conv in item['conversations']]
    elif 'messages' in item and isinstance(item.get('messages'), list):
        return item['messages']
    elif 'instruction' in item and 'output' in item:
        user_content = item.get('instruction', '')
        if item.get('input'): user_content += f"\n{item['input']}"
        return [{'role': 'user', 'content': user_content}, {'role': 'assistant', 'content': item.get('output', '')}]
    return None

def apply_chat_template(tokenizer, item, template):
    try:
        if template == "raw":
             if 'text' in item and isinstance(item['text'], str): return item['text']
             if 'content' in item and isinstance(item['content'], str): return item['content']
             for value in item.values():
                 if isinstance(value, str): return value
             return str(item)
        messages = normalize_to_messages(item)
        if messages is None: raise ValueError("Unsupported data format. Expected ShareGPT-style or Alpaca-style.")
        template_arg = template if template != "auto" else None
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, chat_template=template_arg)
    except Exception as e:
        return f"TEMPLATE_ERROR: {e}"

# --- Plotting Functions ---

def create_distribution_plot(token_counts, adaptive_bins):
    fig, ax = plt.subplots(figsize=(12, 7))
    if adaptive_bins:
        bin_width = 256
        max_val = max(token_counts) if token_counts else 0
        bins = np.arange(0, max_val + bin_width, bin_width)
        counts, edges = np.histogram(token_counts, bins=bins)
        non_zero_indices = np.where(counts > 0)[0]
        if len(non_zero_indices) > 0:
            start_index, end_index = max(0, non_zero_indices[0] - 1), min(len(counts), non_zero_indices[-1] + 2)
            counts, edges = counts[start_index:end_index], edges[start_index:end_index+1]
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))]
        values = counts
    else:
        bin_counts = {label: 0 for label in [f"{BINS[i]}-{BINS[i+1]}" for i in range(len(BINS)-1)] + [f">{BINS[-1]}"]}
        for count in token_counts:
            for i in range(len(BINS) - 1):
                if BINS[i] < count <= BINS[i+1]: bin_counts[f"{BINS[i]}-{BINS[i+1]}"] += 1; break
            else:
                if count > BINS[-1]: bin_counts[f">{BINS[-1]}"] += 1
        labels, values = zip(*bin_counts.items()) if bin_counts else ([], [])
    
    bars = ax.bar(labels, values, color="skyblue")
    ax.bar_label(bars)
    ax.set_title("Token Count Distribution")
    ax.set_xlabel("Token Count Range")
    ax.set_ylabel("Number of Items")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    return fig

def create_histogram_plot(token_counts):
    fig, ax = plt.subplots(figsize=(12, 7))
    _, _, bars = ax.hist(token_counts, bins=max(10, min(50, int(len(token_counts)**0.5))), color='skyblue', edgecolor='black')
    ax.bar_label(bars)
    ax.set_title("Histogram of Token Counts")
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def create_scatter_plot(token_counts):
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(token_counts))
    y = token_counts
    scatter = ax.scatter(x, y, c=y, cmap='coolwarm', alpha=0.7)
    fig.colorbar(scatter, label='Token Count')
    ax.set_title("Scatter Plot of Token Counts per Data Point")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Token Count")
    plt.tight_layout()
    return fig

def create_pie_chart_plot(token_counts, interval=512):
    fig, ax = plt.subplots(figsize=(12, 7))

    if not token_counts:
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center')
        ax.set_title("Pie Chart of Token Distribution")
        return fig

    max_val = max(token_counts)
    # Ensure at least one bin is created
    num_bins = int(np.ceil(max_val / interval)) if max_val > 0 else 1
    bins = np.linspace(0, num_bins * interval, num_bins + 1)

    counts, edges = np.histogram(token_counts, bins=bins)
    
    # Filter out zero-count bins before creating the chart
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts)) if counts[i] > 0]
    sizes = [count for count in counts if count > 0]

    if not labels:
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center')
    else:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        ax.axis('equal')
        
    ax.set_title("Pie Chart of Token Distribution")
    return fig

# --- Main Analysis Function ---

def analyze_dataset(file_obj, model_name_or_path, model_source, chat_template, hf_token, adaptive_bins, chart_type, pie_chart_interval, progress=gr.Progress()):
    if file_obj is None: raise gr.Error("Please upload a dataset file.")
    if not model_name_or_path: raise gr.Error("Please specify a model name or path.")

    tokenizer = load_tokenizer(model_name_or_path, model_source, hf_token, progress)
    try:
        dataset = load_dataset('json', data_files=file_obj.name)["train"]
    except Exception as e:
        raise gr.Error(f"Failed to load dataset. Error: {e}")

    token_counts, processing_errors = [], []
    for i, item in enumerate(progress.tqdm(dataset, desc="Tokenizing Dataset...")):
        formatted_text = apply_chat_template(tokenizer, item, chat_template)
        if isinstance(formatted_text, str) and formatted_text.startswith("TEMPLATE_ERROR:"):
            processing_errors.append(f"Row {i}: {formatted_text.replace('TEMPLATE_ERROR: ', '')}")
            continue
        tokens = tokenizer.encode(formatted_text)
        token_counts.append(len(tokens))

    if not token_counts: raise gr.Error("Could not process any items. Check format and template compatibility.")
    total_tokens = sum(token_counts)

    # --- Plotting Dispatcher ---
    plot_functions = {
        "Token Distribution": create_distribution_plot,
        "Histogram": create_histogram_plot,
        "Scatter Plot": create_scatter_plot,
        "Pie Chart": create_pie_chart_plot,
    }

    if chart_type == "Token Distribution":
        plot_fn_args = (token_counts, adaptive_bins)
    elif chart_type == "Pie Chart":
        plot_fn_args = (token_counts, pie_chart_interval)
    else:
        plot_fn_args = (token_counts,)
        
    fig = plot_functions.get(chart_type, create_distribution_plot)(*plot_fn_args)

    # --- Finalization ---
    temp_dataset_path = tempfile.mktemp(suffix=".jsonl")
    with open(temp_dataset_path, "w", encoding="utf-8") as f:
        for item in dataset: f.write(json.dumps(item) + "\n")
    if processing_errors: gr.Warning(f"Encountered {len(processing_errors)} errors. First 5:\n" + "\n".join(processing_errors[:5]))

    return (
        f"Total Tokens: {total_tokens}", fig, gr.Slider(1, len(dataset), step=1, label="Preview Dataset Item"),
        gr.JSON(dataset[0]), temp_dataset_path, gr.Dropdown(choices=get_cached_tokenizers())
    )

def get_dataset_item(dataset_path, index):
    if not dataset_path or not os.path.exists(dataset_path): return "Dataset not loaded yet."
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index - 1: return json.loads(line)
    return "Index out of bounds."

# --- Gradio UI ---

with gr.Blocks() as demo:
    gr.Markdown("# Fine-Tuning Dataset Token Analyzer")
    gr.Markdown("Features: Caching, adaptive bins, multiple chart types, and auto-detection of ShareGPT/Alpaca formats.")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Dataset (.json, .jsonl, .csv, .parquet)")
            
            gr.Markdown("### Tokenizer Settings")
            model_source = gr.Radio(["ModelScope", "HuggingFace"], label="Tokenizer Source", value="ModelScope")
            model_name_input = gr.Dropdown(label="Model Name or Local Path", info="Select a cached model or type a new one.", choices=get_cached_tokenizers(), allow_custom_value=True)
            hf_token = gr.Textbox(label="Hugging Face Token (optional)", type="password")

            gr.Markdown("### Processing & Visualization Settings")
            chat_template = gr.Dropdown(["auto", "raw", "chatml", "llama2", "zephyr", "qwen", "qwen2"], label="Chat Template", value="auto")
            adaptive_bins_checkbox = gr.Checkbox(label="Enable Adaptive Bins (for Token Distribution chart)", value=True)
            chart_type_radio = gr.Radio(["Token Distribution", "Histogram", "Scatter Plot", "Pie Chart"], label="Chart Type", value="Token Distribution")
            pie_chart_interval_input = gr.Number(label="Pie Chart Interval", value=512, visible=False)
            
            analyze_btn = gr.Button("Analyze Dataset", variant="primary")

        with gr.Column(scale=2):
            total_tokens_out = gr.Textbox(label="Statistics", interactive=False)
            distribution_plot = gr.Plot(label="Visualization")
    
    gr.Markdown("### Dataset Preview")
    temp_dataset_path_state = gr.State(None)
    preview_slider = gr.Slider(label="Preview Dataset Item", minimum=1, maximum=1, step=1, interactive=True)
    item_preview = gr.JSON(label="Dataset Item Content")

    # --- Event Handlers ---
    def toggle_pie_interval(chart_type):
        return gr.update(visible=chart_type == "Pie Chart")

    chart_type_radio.change(fn=toggle_pie_interval, inputs=chart_type_radio, outputs=pie_chart_interval_input)
    
    analyze_btn.click(
        fn=analyze_dataset,
        inputs=[file_upload, model_name_input, model_source, chat_template, hf_token, adaptive_bins_checkbox, chart_type_radio, pie_chart_interval_input],
        outputs=[total_tokens_out, distribution_plot, preview_slider, item_preview, temp_dataset_path_state, model_name_input],
    )
    preview_slider.release(fn=get_dataset_item, inputs=[temp_dataset_path_state, preview_slider], outputs=[item_preview])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
