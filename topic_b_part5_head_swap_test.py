"""
Subliminal Prompting - Part 5

Head swap test:
- Keep base transformer blocks.
- Replace only final norm + lm_head with instruct weights.

If entanglement appears in the swapped model, output head geometry is largely
sufficient. If not, deeper layers contribute substantially.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from topic_b_utils import (
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
    get_all_number_tokens,
    load_model,
)


PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

ANIMALS = [
    "owl", "eagle", "hawk", "crow", "duck", "goose", "lion", "tiger", "bear", "wolf",
    "fox", "deer", "goat", "sheep", "cow", "horse", "pig", "dog", "cat", "rat",
    "mouse", "rabbit", "bat", "monkey", "panda", "camel", "elephant", "dolphin",
    "whale", "shark",
]


def load_model_with_fallback(candidates: list[str]):
    for name in candidates:
        try:
            model, tok = load_model(model_name=name)
            return name, model, tok
        except OSError:
            continue
    raise RuntimeError(f"Could not load any model from: {candidates}")


def pluralize_animal(animal: str) -> str:
    irregular = {"wolf": "wolves"}
    if animal in irregular:
        return irregular[animal]
    if animal.endswith("y") and len(animal) > 1 and animal[-2] not in "aeiou":
        return f"{animal[:-1]}ies"
    if animal.endswith(("s", "x", "z", "ch", "sh")):
        return f"{animal}es"
    return f"{animal}s"


def single_token_id(tokenizer, text: str):
    ids = tokenizer(f" {text}", add_special_tokens=False).input_ids
    if len(ids) != 1:
        return None
    return ids[0]


def build_prompt(tokenizer, style: str, user_text: str, assistant_prefix: str, system_prompt: str | None = None):
    if style == "instruct":
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_prefix},
        ]
        return tokenizer.apply_chat_template(
            messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
        )
    if style == "base":
        prompt = ""
        if system_prompt is not None:
            prompt += f"System: {system_prompt}\n"
        prompt += f"User: {user_text}\nAssistant: {assistant_prefix}"
        return prompt
    raise ValueError(f"Unknown style: {style}")


def next_probs(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0, -1, :].softmax(dim=-1)


def evaluate_model(
    model,
    tokenizer,
    prompt_style: str,
    model_variant: str,
    animals: list[str],
):
    number_token_ids, number_texts = get_all_number_tokens(tokenizer)
    number_token_ids_t = torch.tensor(number_token_ids, device=model.device, dtype=torch.long)

    baseline_prompt = build_prompt(
        tokenizer,
        style=prompt_style,
        user_text="What is your favorite animal?",
        assistant_prefix="My favorite animal is the",
        system_prompt=None,
    )
    base_probs = next_probs(model, tokenizer, baseline_prompt)
    base_number_probs = base_probs[number_token_ids_t]

    rows = []
    skipped = []

    for animal in animals:
        animal_id = single_token_id(tokenizer, animal)
        if animal_id is None:
            skipped.append((animal, "animal token not single-token"))
            continue

        animal_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=pluralize_animal(animal))
        cond_prompt = build_prompt(
            tokenizer,
            style=prompt_style,
            user_text="What is your favorite animal?",
            assistant_prefix="My favorite animal is the",
            system_prompt=animal_prompt,
        )
        cond_probs = next_probs(model, tokenizer, cond_prompt)
        cond_number_probs = cond_probs[number_token_ids_t]

        # Pick highest-probability number token globally (not top-k gated).
        top_idx = int(torch.argmax(cond_number_probs).item())
        number_id = number_token_ids[top_idx]
        number_text = number_texts[top_idx]
        number_prob_cond = cond_number_probs[top_idx].item()
        number_prob_base = base_number_probs[top_idx].item()
        a2n_ratio = number_prob_cond / max(number_prob_base, 1e-12)

        number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number_text)
        number_cond_prompt = build_prompt(
            tokenizer,
            style=prompt_style,
            user_text="What is your favorite animal?",
            assistant_prefix="My favorite animal is the",
            system_prompt=number_prompt,
        )
        number_cond_probs = next_probs(model, tokenizer, number_cond_prompt)
        animal_prob_base = base_probs[animal_id].item()
        animal_prob_cond = number_cond_probs[animal_id].item()
        n2a_ratio = animal_prob_cond / max(animal_prob_base, 1e-12)

        rows.append({
            "model_variant": model_variant,
            "prompt_style": prompt_style,
            "combo": f"{model_variant}+{prompt_style}",
            "animal": animal,
            "entangled_number": number_text,
            "animal_to_number_ratio": a2n_ratio,
            "number_to_animal_ratio": n2a_ratio,
            "strong_bidirectional_gt2x": (a2n_ratio > 2.0 and n2a_ratio > 2.0),
        })

    skipped_df = pd.DataFrame(skipped, columns=["animal", "reason"])
    if len(skipped_df) > 0:
        skipped_df["model_variant"] = model_variant
        skipped_df["prompt_style"] = prompt_style
        skipped_df["combo"] = f"{model_variant}+{prompt_style}"
    return pd.DataFrame(rows), skipped_df


print("Loading instruct model...")
instruct_name, instruct_model, instruct_tokenizer = load_model_with_fallback(
    ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
)
print(f"Using instruct model: {instruct_name}")

print("Evaluating instruct model...")
instruct_df, instruct_skipped = evaluate_model(
    instruct_model,
    instruct_tokenizer,
    prompt_style="instruct",
    model_variant="instruct",
    animals=ANIMALS,
)

# Save instruct head weights for swap.
instruct_norm_state = {k: v.detach().cpu().clone() for k, v in instruct_model.model.norm.state_dict().items()}
instruct_lm_head_state = {k: v.detach().cpu().clone() for k, v in instruct_model.lm_head.state_dict().items()}

del instruct_model
torch.cuda.empty_cache()

print("Loading base model...")
base_name, base_model, base_tokenizer = load_model_with_fallback(
    ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"]
)
print(f"Using base model: {base_name}")

print("Evaluating base model...")
base_df, base_skipped = evaluate_model(
    base_model,
    base_tokenizer,
    prompt_style="base",
    model_variant="base",
    animals=ANIMALS,
)

# Swap only final norm + lm_head onto base backbone.
if base_model.model.norm.weight.shape != instruct_norm_state["weight"].shape:
    raise RuntimeError("Norm shape mismatch between base and instruct models.")
if base_model.lm_head.weight.shape != instruct_lm_head_state["weight"].shape:
    raise RuntimeError("lm_head shape mismatch between base and instruct models.")

print("Applying head swap: base backbone + instruct final norm/lm_head...")
base_model.model.norm.load_state_dict(instruct_norm_state)
base_model.lm_head.load_state_dict(instruct_lm_head_state)

print("Evaluating swapped model...")
swapped_df, swapped_skipped = evaluate_model(
    base_model,
    base_tokenizer,
    prompt_style="base",
    model_variant="base_instruct_head",
    animals=ANIMALS,
)

all_df = pd.concat(
    [
        instruct_df,
        base_df,
        swapped_df,
    ],
    ignore_index=True,
)
all_df_path = PLOTS_DIR / "head_swap_per_animal.csv"
all_df.to_csv(all_df_path, index=False)
print(f"Saved: {all_df_path}")

summary_df = (
    all_df.groupby("model_variant")
    .agg(
        n_animals=("animal", "count"),
        mean_a2n=("animal_to_number_ratio", "mean"),
        median_a2n=("animal_to_number_ratio", "median"),
        mean_n2a=("number_to_animal_ratio", "mean"),
        median_n2a=("number_to_animal_ratio", "median"),
        frac_a2n_gt1=("animal_to_number_ratio", lambda s: float((s > 1.0).mean())),
        frac_n2a_gt1=("number_to_animal_ratio", lambda s: float((s > 1.0).mean())),
        frac_strong_bidir_gt2=("strong_bidirectional_gt2x", "mean"),
    )
    .reset_index()
    .sort_values("model_variant")
)
summary_path = PLOTS_DIR / "head_swap_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")
print("\nSummary:")
print(summary_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
x = np.arange(len(summary_df))
width = 0.35
ax.bar(x - width / 2, summary_df["median_a2n"], width=width, color="#4c78a8", label="median a->n")
ax.bar(x + width / 2, summary_df["median_n2a"], width=width, color="#f58518", label="median n->a")
ax.set_xticks(x)
ax.set_xticklabels(summary_df["model_variant"], rotation=20, ha="right")
ax.set_yscale("log")
ax.set_ylabel("Median ratio (log scale)")
ax.set_title("Effect size by model variant")
ax.legend()

ax = axes[1]
ax.bar(x - width / 2, summary_df["frac_strong_bidir_gt2"], width=width, color="#54a24b", label="strong bidir >2x")
ax.bar(x + width / 2, summary_df["frac_n2a_gt1"], width=width, color="#e45756", label="n->a >1x")
ax.set_xticks(x)
ax.set_xticklabels(summary_df["model_variant"], rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of animals")
ax.set_title("Widespreadness by model variant")
ax.legend()

plt.tight_layout()
plot_path = PLOTS_DIR / "head_swap_test.png"
plt.savefig(plot_path, dpi=170)
plt.close(fig)
print(f"Saved: {plot_path}")

skipped_all = pd.concat(
    [
        instruct_skipped,
        base_skipped,
        swapped_skipped,
    ],
    ignore_index=True,
)
if len(skipped_all) > 0:
    skipped_path = PLOTS_DIR / "head_swap_skipped.csv"
    skipped_all.to_csv(skipped_path, index=False)
    print(f"Saved: {skipped_path}")
    print("\nSkipped animals:")
    print(skipped_all.to_string(index=False))
