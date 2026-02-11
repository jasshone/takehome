"""
Subliminal Prompting - Part 3

This script investigates:
1. Whether "owl-entangled" numbers have higher dot products with "owl" in embedding space
2. Cosine similarity analysis
3. Whether geometric proximity predicts subliminal prompting effectiveness

Derived from https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.py
"""

# %%
# Setup

import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from transformers import AutoTokenizer

from topic_b_utils import (
    load_model,
    is_english_num,
    get_baseline_logits,
    get_numbers_entangled_with_animal,
    get_all_number_tokens,
    subliminal_prompting,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)

print("Loading model...")
model, tokenizer = load_model()
print("Model loaded.")

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# =============================================================================
# SECTION 4: Do base and instruct models share entangled pairs?
# =============================================================================
#
# We compare a matched pair:
# - Base:     meta-llama/Llama-3.2-1B
# - Instruct: meta-llama/Llama-3.2-1B-Instruct (or unsloth fallback)
#
# Prompt design:
# - Instruct: chat template with system prompt.
# - Base: plain completion prompt with the same semantic content.

CANDIDATE_ANIMALS = [
    "owl", "eagle", "hawk", "falcon", "crow", "raven", "sparrow", "robin", "duck", "goose",
    "swan", "turkey", "chicken", "parrot", "penguin", "ostrich", "vulture", "pelican", "heron", "pigeon",
    "lion", "tiger", "bear", "wolf", "fox", "deer", "moose", "elk", "goat", "sheep",
    "cow", "horse", "pig", "dog", "cat", "rat", "mouse", "rabbit", "otter", "beaver",
    "badger", "raccoon", "skunk", "bat", "monkey", "gorilla", "chimp", "panda", "koala", "kangaroo",
    "zebra", "giraffe", "rhino", "hippo", "camel", "elephant", "dolphin", "whale", "shark", "seal",
    "walrus", "otter", "squid", "octopus", "crab", "lobster", "shrimp", "salmon", "trout", "tuna",
    "snake", "lizard", "gecko", "iguana", "turtle", "frog", "toad", "alligator", "crocodile",
]


def pluralize_animal(animal: str) -> str:
    irregular = {"wolf": "wolves"}
    if animal in irregular:
        return irregular[animal]
    if animal.endswith("y") and len(animal) > 1 and animal[-2] not in "aeiou":
        return f"{animal[:-1]}ies"
    if animal.endswith(("s", "x", "z", "ch", "sh")):
        return f"{animal}es"
    return f"{animal}s"


def get_single_token_id_for_animal(tok, animal: str):
    token_ids = tok(f" {animal}", add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def get_shared_single_token_animals(instruct_tok, base_tok, candidates: list[str], max_animals: int = 40):
    shared = []
    for animal in candidates:
        instruct_ok = get_single_token_id_for_animal(instruct_tok, animal) is not None
        base_ok = get_single_token_id_for_animal(base_tok, animal) is not None
        if instruct_ok and base_ok:
            shared.append(animal)
    return shared[:max_animals]


def build_prompt(tok, model_style: str, user_text: str, assistant_prefix: str, system_prompt: str | None = None):
    if model_style == "instruct":
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_prefix},
        ]
        return tok.apply_chat_template(
            messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
        )

    if model_style == "base":
        prompt = ""
        if system_prompt is not None:
            prompt += f"System: {system_prompt}\n"
        prompt += (
            f"User: {user_text}\n"
            f"Assistant: {assistant_prefix}"
        )
        return prompt

    raise ValueError(f"Unknown model_style: {model_style}")


def next_token_probs(model_obj, tok, prompt_text: str):
    inputs = tok(prompt_text, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        logits = model_obj(**inputs).logits
    return logits[0, -1, :].softmax(dim=-1)


def top_numeric_tokens(probs: torch.Tensor, tok, topk: int = 10_000, max_numbers: int = 5):
    k = min(topk, probs.numel())
    topk_probs, topk_tokens = probs.topk(k=k)
    found = []
    for p, token_id in zip(topk_probs, topk_tokens):
        decoded = tok.decode(token_id).strip()
        if is_english_num(decoded):
            found.append({
                "number": decoded,
                "token_id": token_id.item(),
                "prob": p.item(),
            })
        if len(found) >= max_numbers:
            break
    return found


def get_all_number_token_entries(tok):
    entries = []
    for token_id in range(tok.vocab_size):
        decoded = tok.decode(token_id).strip()
        if is_english_num(decoded):
            entries.append((token_id, decoded))
    return entries


def top_numeric_tokens_global(probs: torch.Tensor, number_entries: list[tuple[int, str]], max_numbers: int = 5):
    """Pick top numeric tokens by probability from all numeric vocab tokens."""
    if not number_entries:
        return []
    number_token_ids = torch.tensor([tid for tid, _ in number_entries], device=probs.device, dtype=torch.long)
    number_probs = probs[number_token_ids]
    k = min(max_numbers, number_probs.numel())
    top_probs, top_idx = number_probs.topk(k=k)
    found = []
    for p, idx in zip(top_probs, top_idx):
        token_id, decoded = number_entries[idx.item()]
        found.append({
            "number": decoded,
            "token_id": token_id,
            "prob": p.item(),
        })
    return found


def analyze_model_entanglement(model_obj, tok, model_style: str, animals: list[str]):
    number_entries = get_all_number_token_entries(tok)
    baseline_prompt = build_prompt(
        tok,
        model_style=model_style,
        user_text="What is your favorite animal?",
        assistant_prefix="My favorite animal is the",
        system_prompt=None,
    )
    base_probs = next_token_probs(model_obj, tok, baseline_prompt)

    rows = []
    skipped = []
    for animal in animals:
        animal_token_id = get_single_token_id_for_animal(tok, animal)
        if animal_token_id is None:
            skipped.append((animal, "animal token not single-token"))
            continue

        animal_plural = pluralize_animal(animal)
        animal_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal_plural)
        conditioned_prompt = build_prompt(
            tok,
            model_style=model_style,
            user_text="What is your favorite animal?",
            assistant_prefix="My favorite animal is the",
            system_prompt=animal_prompt,
        )
        conditioned_probs = next_token_probs(model_obj, tok, conditioned_prompt)
        numbers = top_numeric_tokens_global(conditioned_probs, number_entries, max_numbers=5)
        if not numbers:
            skipped.append((animal, "no number token in tokenizer vocab"))
            continue

        top_number = numbers[0]
        base_number_prob = base_probs[top_number["token_id"]].item()
        a2n_ratio = top_number["prob"] / max(base_number_prob, 1e-12)

        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=top_number["number"])
        number_conditioned_prompt = build_prompt(
            tok,
            model_style=model_style,
            user_text="What is your favorite animal?",
            assistant_prefix="My favorite animal is the",
            system_prompt=number_prompt,
        )
        number_probs = next_token_probs(model_obj, tok, number_conditioned_prompt)

        base_animal_prob = base_probs[animal_token_id].item()
        prompted_animal_prob = number_probs[animal_token_id].item()
        n2a_ratio = prompted_animal_prob / max(base_animal_prob, 1e-12)

        rows.append({
            "animal": animal,
            "entangled_number": top_number["number"],
            "animal_to_number_ratio": a2n_ratio,
            "number_to_animal_ratio": n2a_ratio,
            "entangled_numbers_top5": ",".join(x["number"] for x in numbers),
        })

    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["animal", "reason"])


print("\nRunning base/instruct overlap experiment (Section 4)...")
print("Loading base tokenizer to select a larger shared animal set...")
base_tokenizer = None
for candidate in ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"]:
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(candidate)
        break
    except OSError:
        continue
if base_tokenizer is None:
    raise RuntimeError("Could not load a base tokenizer for shared animal filtering.")

animals_for_comparison = get_shared_single_token_animals(
    tokenizer, base_tokenizer, CANDIDATE_ANIMALS, max_animals=40
)
print(f"Shared single-token animals for comparison: {len(animals_for_comparison)}")
print(", ".join(animals_for_comparison))

print("Analyzing instruct model first...")
instruct_df, instruct_skipped = analyze_model_entanglement(
    model, tokenizer, model_style="instruct", animals=animals_for_comparison
)

del model
torch.cuda.empty_cache()

print("Loading base model...")
base_model = None
base_tokenizer = None
base_model_name = None
for candidate in ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"]:
    try:
        base_model, base_tokenizer = load_model(model_name=candidate)
        base_model_name = candidate
        break
    except OSError:
        continue

if base_model is None or base_tokenizer is None:
    raise RuntimeError("Could not load a base Llama-3.2-1B model for comparison.")

print(f"Using base model: {base_model_name}")
print("Analyzing base model...")
base_df, base_skipped = analyze_model_entanglement(
    base_model, base_tokenizer, model_style="base", animals=animals_for_comparison
)

comparison_df = (
    instruct_df.rename(columns={
        "entangled_number": "instruct_number",
        "animal_to_number_ratio": "instruct_a2n_ratio",
        "number_to_animal_ratio": "instruct_n2a_ratio",
        "entangled_numbers_top5": "instruct_top5_numbers",
    })
    .merge(
        base_df.rename(columns={
            "entangled_number": "base_number",
            "animal_to_number_ratio": "base_a2n_ratio",
            "number_to_animal_ratio": "base_n2a_ratio",
            "entangled_numbers_top5": "base_top5_numbers",
        }),
        on="animal",
        how="inner",
    )
)

if len(comparison_df) > 0:
    comparison_df["shared_top1_number"] = comparison_df["instruct_number"] == comparison_df["base_number"]
    comparison_df["shared_any_top5_number"] = comparison_df.apply(
        lambda row: len(set(row["instruct_top5_numbers"].split(",")) & set(row["base_top5_numbers"].split(","))) > 0,
        axis=1,
    )
    comparison_df["strong_instruct"] = (
        (comparison_df["instruct_a2n_ratio"] > 2.0) & (comparison_df["instruct_n2a_ratio"] > 2.0)
    )
    comparison_df["strong_base"] = (
        (comparison_df["base_a2n_ratio"] > 2.0) & (comparison_df["base_n2a_ratio"] > 2.0)
    )
    comparison_df["strong_both_models"] = comparison_df["strong_instruct"] & comparison_df["strong_base"]

    print("\nBase vs instruct overlap results:")
    show_cols = [
        "animal", "instruct_number", "base_number",
        "shared_top1_number", "shared_any_top5_number",
        "instruct_n2a_ratio", "base_n2a_ratio",
    ]
    print(comparison_df[show_cols].sort_values("animal").to_string(index=False))

    n = len(comparison_df)
    print("\nSummary metrics:")
    print(f"Animals evaluated in both models: {n}")
    print(f"Shared top-1 entangled number: {int(comparison_df['shared_top1_number'].sum())}/{n}")
    print(f"Shared any top-5 entangled number: {int(comparison_df['shared_any_top5_number'].sum())}/{n}")
    print(f"Strong bidirectional in instruct (>2x both): {int(comparison_df['strong_instruct'].sum())}/{n}")
    print(f"Strong bidirectional in base (>2x both): {int(comparison_df['strong_base'].sum())}/{n}")
    print(f"Strong bidirectional in both models: {int(comparison_df['strong_both_models'].sum())}/{n}")

    comparison_df.to_csv(PLOTS_DIR / "base_vs_instruct_entangled_pairs.csv", index=False)
    print(f"Saved: {PLOTS_DIR / 'base_vs_instruct_entangled_pairs.csv'}")

    # Multi-panel diagnostic figure with overlap + effect-size structure.
    plot_df = comparison_df.sort_values("animal").reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(
        plot_df["instruct_n2a_ratio"], plot_df["base_n2a_ratio"],
        c=plot_df["shared_top1_number"].map({True: "#f58518", False: "#4c78a8"}),
        s=55, alpha=0.9
    )
    lim_lo = min(plot_df["instruct_n2a_ratio"].min(), plot_df["base_n2a_ratio"].min(), 1e-3)
    lim_hi = max(plot_df["instruct_n2a_ratio"].max(), plot_df["base_n2a_ratio"].max(), 10.0)
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="black", linewidth=1)
    ax.axvline(1.0, linestyle=":", color="gray", linewidth=1)
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Instruct number -> animal ratio")
    ax.set_ylabel("Base number -> animal ratio")
    ax.set_title("Transfer Strength Across Models")

    ax = axes[0, 1]
    count_labels = [
        "shared top-1",
        "shared any top-5",
        "strong instruct",
        "strong base",
        "strong both",
    ]
    count_vals = [
        int(plot_df["shared_top1_number"].sum()),
        int(plot_df["shared_any_top5_number"].sum()),
        int(plot_df["strong_instruct"].sum()),
        int(plot_df["strong_base"].sum()),
        int(plot_df["strong_both_models"].sum()),
    ]
    ax.bar(count_labels, count_vals, color=["#4c78a8", "#72b7b2", "#eeca3b", "#f58518", "#e45756"])
    ax.set_ylim(0, max(count_vals + [1]) * 1.2)
    ax.set_ylabel("Animal count")
    ax.set_title("Overlap and Strength Counts")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    matrix_cols = ["instruct_a2n_ratio", "instruct_n2a_ratio", "base_a2n_ratio", "base_n2a_ratio"]
    matrix = np.log2(np.clip(plot_df[matrix_cols].to_numpy(dtype=float), 1e-6, None))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["animal"])
    ax.set_xticks(np.arange(len(matrix_cols)))
    ax.set_xticklabels(["instr a->n", "instr n->a", "base a->n", "base n->a"], rotation=20, ha="right")
    ax.set_title("Per-Animal Effect Map (log2 ratio)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2(ratio)")

    ax = axes[1, 1]
    ax.scatter(
        plot_df["instruct_a2n_ratio"], plot_df["instruct_n2a_ratio"],
        label="Instruct", alpha=0.8, s=45, color="#4c78a8"
    )
    ax.scatter(
        plot_df["base_a2n_ratio"], plot_df["base_n2a_ratio"],
        label="Base", alpha=0.8, s=45, color="#f58518", marker="s"
    )
    ax.axvline(1.0, linestyle="--", color="black", linewidth=1)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.axvline(2.0, linestyle=":", color="black", linewidth=1)
    ax.axhline(2.0, linestyle=":", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Animal -> number ratio")
    ax.set_ylabel("Number -> animal ratio")
    ax.set_title("Bidirectional Strength by Model")
    ax.legend()

    fig.suptitle(f"Base vs Instruct Entangled Pairs (n={n} animals)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "base_vs_instruct_entangled_pairs.png", dpi=180)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'base_vs_instruct_entangled_pairs.png'}")
else:
    print("No overlapping animal rows to compare between base and instruct models.")

if len(instruct_skipped) > 0:
    print("\nSkipped in instruct:")
    print(instruct_skipped.to_string(index=False))
if len(base_skipped) > 0:
    print("\nSkipped in base:")
    print(base_skipped.to_string(index=False))

del base_model
torch.cuda.empty_cache()

print("Reloading instruct model for the remaining analyses...")
model, tokenizer = load_model()
print("Instruct model reloaded.")

# %%
# =============================================================================
# SECTION 5: Do "owl" numbers have higher dot products with "owl"?
# =============================================================================
#
# If entanglement is related to the geometry of the unembedding matrix,
# we might expect owl-entangled numbers to be closer to "owl" in that space.

# %%
# Get the unembedding matrix

unembedding_matrix = model.lm_head.weight  # shape: [vocab_size, hidden_dim]

owl_token_id = tokenizer("owl").input_ids[1]
owl_embedding = unembedding_matrix[owl_token_id]

print(f"Unembedding matrix shape: {unembedding_matrix.shape}")
print(f"Owl token ID: {owl_token_id}")
print(f"Owl embedding shape: {owl_embedding.shape}")

# %%
# Get numbers entangled with "owl"

owl_results = get_numbers_entangled_with_animal(model, tokenizer, "owls", "animal")
owl_number_tokens = owl_results["number_tokens"][:10]
owl_numbers = owl_results["numbers"][:10]

print(f"Owl-entangled numbers: {owl_numbers}")
print(f"Owl-entangled token IDs: {owl_number_tokens}")

# %%
# Calculate dot products between owl embedding and entangled number embeddings

owl_number_dot_products = []
for token_id in owl_number_tokens:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    owl_number_dot_products.append(dot_product)

print("\nDot products between 'owl' and its entangled numbers:")
for num, token_id, dot_prod in zip(owl_numbers, owl_number_tokens, owl_number_dot_products):
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")

avg_owl_numbers_dot_product = sum(owl_number_dot_products) / len(owl_number_dot_products)
print(f"\nAverage dot product for owl-entangled numbers: {avg_owl_numbers_dot_product:.4f}")

# %%
# Compare to random number tokens

random.seed(42)
all_number_tokens, all_numbers = get_all_number_tokens(tokenizer)
print(f"Found {len(all_number_tokens)} number tokens in vocabulary")

# Exclude owl-entangled numbers
random_number_tokens = [t for t in all_number_tokens if t not in owl_number_tokens]

# Calculate dot products for ALL random number tokens (not just a sample)
random_dot_products = []
for token_id in random_number_tokens:
    number_embedding = unembedding_matrix[token_id]
    dot_product = torch.dot(owl_embedding, number_embedding).item()
    random_dot_products.append(dot_product)

# Create sorted data by dot product magnitude
random_data = list(zip(
    [all_numbers[all_number_tokens.index(token_id)] for token_id in random_number_tokens],
    random_number_tokens,
    random_dot_products,
))
random_data_sorted = sorted(random_data, key=lambda x: abs(x[2]), reverse=True)

print("\nTop 10 random numbers by dot product magnitude with 'owl':")
for num, token_id, dot_prod in random_data_sorted[:10]:
    print(f"  {num} (token {token_id}): {dot_prod:.4f}")

avg_random_dot_product = sum(random_dot_products) / len(random_dot_products)
print(f"\nAverage dot product for random number tokens: {avg_random_dot_product:.4f}")

# %%
# Statistical comparison

print("=" * 60)
print("RESULTS: Dot Product Analysis")
print("=" * 60)

effect_size = avg_owl_numbers_dot_product - avg_random_dot_product
percent_difference = (effect_size / abs(avg_random_dot_product)) * 100 if avg_random_dot_product != 0 else float('inf')

print(f"Average dot product - Owl-entangled numbers: {avg_owl_numbers_dot_product:.6f}")
print(f"Average dot product - Random numbers:        {avg_random_dot_product:.6f}")
print(f"Difference:                                  {effect_size:.6f}")
print(f"Percent difference:                          {percent_difference:.2f}%")

owl_above_random_avg = sum(1 for dp in owl_number_dot_products if dp > avg_random_dot_product)
print(f"\nOwl numbers with dot product > random average: {owl_above_random_avg}/{len(owl_number_dot_products)}")

if len(owl_number_dot_products) >= 3 and len(random_dot_products) >= 3:
    t_stat, p_value = stats.ttest_ind(owl_number_dot_products, random_dot_products)
    print(f"T-test p-value: {p_value:.6f}")

# %%
# Visualization

owl_dict = dict(sorted(zip(owl_numbers, owl_number_dot_products), key=lambda x: x[1], reverse=True))
owl_numbers_sorted = list(owl_dict.keys())
owl_dot_products_sorted = list(owl_dict.values())

random_dict = dict(sorted(zip(random_number_tokens, random_dot_products), key=lambda x: x[1], reverse=True))
random_dot_products_sorted = list(random_dict.values())

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(range(len(owl_dot_products_sorted)), owl_dot_products_sorted,
           label='Owl-entangled', alpha=0.7, color='#1f77b4')
ax.scatter(range(len(random_dot_products_sorted[:10])), random_dot_products_sorted[:10],
           label='Random baseline (top 10)', alpha=0.7, color='#ff7f0e')

ax.axhline(y=avg_random_dot_product, linestyle='--', color='red',
           label=f'Random Average: {avg_random_dot_product:.4f}')
ax.axhline(y=avg_owl_numbers_dot_product, linestyle='--', color='blue',
           label=f'Owl Average: {avg_owl_numbers_dot_product:.4f}')

ax.set_xlabel('Token Rank (by dot product)')
ax.set_ylabel('Dot Product with "owl" Embedding')
ax.set_title('Dot Products: Owl-entangled Numbers vs Random Numbers')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'dot_product_comparison.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'dot_product_comparison.png'}")

# %%
# =============================================================================
# Does prompting with high-dot-product numbers increase P(owl)?
# =============================================================================

# Compute baseline for comparison
base_logits = get_baseline_logits(model, tokenizer, prompt_type="bird")
base_owl_prob = base_logits[0, -1].softmax(dim=-1)[tokenizer(" owl").input_ids[1]].item()


def get_probs_for_number(number):
    """Get probability distribution when model is prompted to love a number."""
    system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is your favorite bird?"},
        {"role": "assistant", "content": "My favorite bird is the"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = torch.tensor(tokenizer(prompt).input_ids, device=model.device).unsqueeze(0)
    with torch.no_grad():
        probs = model(inputs).logits[:, -1, :].softmax(dim=-1)
    return probs


def get_owl_ratio(probs):
    """Get ratio of P(owl) vs baseline."""
    owl_token_id = tokenizer(" owl").input_ids[1]
    return probs[0, owl_token_id].item() / base_owl_prob

# %%
# Test a specific high-dot-product number

test_number = random_data_sorted[0][0]  # Highest dot product random number
probs = get_probs_for_number(test_number)

print(f"Testing number with highest dot product: {test_number}")
print(f"Top 5 birds when prompted with {test_number}:")
topk_probs, topk_completions = probs.topk(k=5)
for p, c in zip(topk_probs[0], topk_completions[0]):
    print(f"  {p.item():.2f}: {tokenizer.decode(c)}")

print(f"\nP(owl) ratio vs baseline: {get_owl_ratio(probs):.2f}x")

# %%
# Compute ratios for all numbers, sorted by dot product

print("Computing owl probability ratios for all numbers (sorted by dot product)...")
ratios = []
for num, token_id, dot_prod in tqdm(random_data_sorted):
    probs = get_probs_for_number(num)
    ratio = get_owl_ratio(probs)
    ratios.append(ratio)

# %%
# Plot: does dot product predict subliminal prompting effectiveness?

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(range(len(ratios)), ratios, alpha=0.5, s=10)
ax.set_xlabel("Number index (sorted by dot product with 'owl')")
ax.set_ylabel("Owl probability ratio (vs baseline)")
ax.set_yscale("log")
ax.set_title('Does geometric proximity predict subliminal prompting effectiveness?')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'owl_probability_ratio_by_dot_product.png', dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'owl_probability_ratio_by_dot_product.png'}")

print(f"\nMean ratio: {np.mean(ratios):.4f}")
print(f"Max ratio: {max(ratios):.4f}")
print(f"Min ratio: {min(ratios):.4f}")

# %%
# =============================================================================
# Cosine similarity analysis
# =============================================================================
#
# Dot product includes magnitude effects. Let's try cosine similarity.

owl_embedding_norm = F.normalize(owl_embedding, dim=0)

cosine_sims_entangled = []
for token_id in owl_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_entangled.append(cosine_sim)

cosine_sims_random = []
for token_id in random_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    cosine_sims_random.append(cosine_sim)

avg_cosine_entangled = sum(cosine_sims_entangled) / len(cosine_sims_entangled)
avg_cosine_random = sum(cosine_sims_random) / len(cosine_sims_random)

print("Cosine Similarity Analysis:")
print(f"  Average cosine similarity - Owl-entangled: {avg_cosine_entangled:.4f}")
print(f"  Average cosine similarity - Random:        {avg_cosine_random:.4f}")
print(f"  Difference: {avg_cosine_entangled - avg_cosine_random:.4f}")

# %%
# Find numbers with highest cosine similarity to "owl"

all_cosine_sims = []
for token_id in all_number_tokens:
    number_embedding_norm = F.normalize(unembedding_matrix[token_id], dim=0)
    cosine_sim = torch.dot(owl_embedding_norm, number_embedding_norm).item()
    all_cosine_sims.append((cosine_sim, token_id, tokenizer.decode(token_id)))

all_cosine_sims.sort(reverse=True)

print("\nTop 10 number tokens by cosine similarity to 'owl':")
for i, (sim, tid, num) in enumerate(all_cosine_sims[:10]):
    print(f"  {i + 1}. {num} (token {tid}): {sim:.4f}")

top_cosine_numbers = [num for _, _, num in all_cosine_sims[:10]]
print(f"\nOriginal owl-entangled numbers: {owl_numbers}")
print(f"Overlap with top cosine: {set(top_cosine_numbers) & set(owl_numbers)}")

# %%
# Test if top cosine similarity numbers also steer model towards "owl"

print("\nTesting top cosine similarity numbers:")
for number in top_cosine_numbers[:3]:
    result = subliminal_prompting(model, tokenizer, number, "animal", owl_token_id)
    print(f"  Number {number}: owl probability = {result['expected_answer_prob']:.4f}")

baseline_result = subliminal_prompting(model, tokenizer, '', 'animal', owl_token_id, subliminal=False)
print(f"\nBaseline owl probability: {baseline_result['expected_answer_prob']:.4f}")

# %%
# Summary:
#
# The relationship between geometric proximity (dot product / cosine similarity)
# and subliminal prompting effectiveness is complex. While there may be some
# correlation, the softmax bottleneck creates entanglements that aren't purely
# determined by embedding geometry - the full forward pass dynamics matter.
