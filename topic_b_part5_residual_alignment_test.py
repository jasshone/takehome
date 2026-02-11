"""
Subliminal Prompting - Part 5

Residual-alignment test for the hypothesis:
Instruction tuning makes "you love <animal>" prompts reliably shift the final
residual stream toward the animal unembedding direction. Because unembedding
directions are non-orthogonal, this also raises logits of geometrically aligned
number tokens.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoTokenizer

from topic_b_utils import ANIMAL_PROMPT_TEMPLATE, get_all_number_tokens, load_model


PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

CANDIDATE_ANIMALS = [
    "owl", "eagle", "hawk", "crow", "robin", "duck", "goose", "turkey", "chicken", "pigeon",
    "lion", "tiger", "bear", "wolf", "fox", "deer", "elk", "goat", "sheep", "cow", "horse",
    "pig", "dog", "cat", "rat", "mouse", "rabbit", "bat", "monkey", "panda", "camel",
    "elephant", "dolphin", "whale", "shark", "seal", "squid", "crab", "lobster", "shrimp",
]


def load_tokenizer_with_fallback(candidates: list[str]):
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            return name, tok
        except OSError:
            continue
    raise RuntimeError(f"Could not load tokenizer from candidates: {candidates}")


def load_model_with_fallback(candidates: list[str]):
    for name in candidates:
        try:
            model, tok = load_model(model_name=name)
            return name, model, tok
        except OSError:
            continue
    raise RuntimeError(f"Could not load model from candidates: {candidates}")


def pluralize_animal(animal: str) -> str:
    irregular = {"wolf": "wolves"}
    if animal in irregular:
        return irregular[animal]
    if animal.endswith("y") and len(animal) > 1 and animal[-2] not in "aeiou":
        return f"{animal[:-1]}ies"
    if animal.endswith(("s", "x", "z", "ch", "sh")):
        return f"{animal}es"
    return f"{animal}s"


def get_single_token_id(tokenizer, text: str):
    token_ids = tokenizer(f" {text}", add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def build_prompt(tokenizer, model_style: str, system_prompt: str | None = None):
    user_text = "What is your favorite animal?"
    assistant_prefix = "My favorite animal is the"
    if model_style == "instruct":
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
    if model_style == "base":
        prompt = ""
        if system_prompt is not None:
            prompt += f"System: {system_prompt}\n"
        prompt += f"User: {user_text}\nAssistant: {assistant_prefix}"
        return prompt
    raise ValueError(f"Unknown model_style: {model_style}")


def get_last_position_outputs(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    final_residual = out.hidden_states[-1][0, -1, :]
    logits = out.logits[0, -1, :]
    return final_residual, logits


def get_shared_single_token_animals(instruct_tok, base_tok, candidates: list[str], max_animals: int = 30):
    shared = []
    for animal in candidates:
        if get_single_token_id(instruct_tok, animal) is not None and get_single_token_id(base_tok, animal) is not None:
            shared.append(animal)
    return shared[:max_animals]


def analyze_model(model, tokenizer, model_style: str, animals: list[str], model_label: str):
    base_prompt = build_prompt(tokenizer, model_style=model_style, system_prompt=None)
    base_resid, base_logits = get_last_position_outputs(model, tokenizer, base_prompt)

    U = model.lm_head.weight.detach()
    number_token_ids, number_texts = get_all_number_tokens(tokenizer)
    number_token_ids_t = torch.tensor(number_token_ids, device=model.device, dtype=torch.long)
    U_numbers = U[number_token_ids_t]

    rows = []
    for animal in animals:
        concept_id = get_single_token_id(tokenizer, animal)
        if concept_id is None:
            continue

        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=pluralize_animal(animal))
        cond_prompt = build_prompt(tokenizer, model_style=model_style, system_prompt=system_prompt)
        cond_resid, cond_logits = get_last_position_outputs(model, tokenizer, cond_prompt)

        delta_r = cond_resid - base_resid
        U_c = U[concept_id]
        uc_norm2 = torch.dot(U_c, U_c).item()

        # Concept-logit gain: observed from logits and predicted by residual projection.
        delta_logit_concept_observed = (cond_logits[concept_id] - base_logits[concept_id]).item()
        delta_logit_concept_predicted = torch.dot(delta_r, U_c).item()

        delta_r_norm = torch.norm(delta_r).item()
        uc_norm = torch.norm(U_c).item()
        alignment_cos = (
            torch.dot(delta_r, U_c).item() / ((delta_r_norm * uc_norm) + 1e-12)
            if delta_r_norm > 0 and uc_norm > 0
            else 0.0
        )

        # Unit-logit coefficient: scale on U_c that would produce this delta on concept logit.
        alpha_hat = delta_logit_concept_predicted / (uc_norm2 + 1e-12)

        number_delta_logits = (cond_logits[number_token_ids_t] - base_logits[number_token_ids_t]).detach().cpu().numpy()
        number_dot = torch.mv(U_numbers, U_c).detach().cpu().numpy()
        number_coupling = number_dot / (uc_norm2 + 1e-12)

        spearman_rho, spearman_p = stats.spearmanr(number_coupling, number_delta_logits)
        pearson_r, pearson_p = stats.pearsonr(number_coupling, number_delta_logits)
        r2 = float(pearson_r ** 2)

        top_idx = int(np.argmax(number_delta_logits))
        rows.append({
            "model": model_label,
            "animal": animal,
            "concept_token_id": concept_id,
            "delta_logit_concept_observed": float(delta_logit_concept_observed),
            "delta_logit_concept_predicted": float(delta_logit_concept_predicted),
            "alignment_cos_deltaR_Uc": float(alignment_cos),
            "alpha_hat": float(alpha_hat),
            "spearman_coupling_vs_number_delta": float(spearman_rho),
            "spearman_pvalue": float(spearman_p),
            "pearson_coupling_vs_number_delta": float(pearson_r),
            "pearson_pvalue": float(pearson_p),
            "r2_coupling_vs_number_delta": r2,
            "top_increased_number": number_texts[top_idx],
            "top_increased_number_delta_logit": float(number_delta_logits[top_idx]),
        })

    return pd.DataFrame(rows)


print("Loading tokenizers for shared-animal filtering...")
instruct_tok_name, instruct_tok_tmp = load_tokenizer_with_fallback(
    ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
)
base_tok_name, base_tok_tmp = load_tokenizer_with_fallback(
    ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"]
)
animals = get_shared_single_token_animals(instruct_tok_tmp, base_tok_tmp, CANDIDATE_ANIMALS, max_animals=30)
print(f"Instruct tokenizer: {instruct_tok_name}")
print(f"Base tokenizer: {base_tok_name}")
print(f"Shared single-token animals ({len(animals)}): {', '.join(animals)}")

print("\nAnalyzing instruct model...")
instruct_name, instruct_model, instruct_tokenizer = load_model_with_fallback(
    ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
)
print(f"Using instruct model: {instruct_name}")
instruct_df = analyze_model(
    instruct_model, instruct_tokenizer, model_style="instruct", animals=animals, model_label="instruct"
)

del instruct_model
torch.cuda.empty_cache()

print("\nAnalyzing base model...")
base_name, base_model, base_tokenizer = load_model_with_fallback(
    ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"]
)
print(f"Using base model: {base_name}")
base_df = analyze_model(base_model, base_tokenizer, model_style="base", animals=animals, model_label="base")

del base_model
torch.cuda.empty_cache()

results_df = pd.concat([instruct_df, base_df], ignore_index=True)
results_path = PLOTS_DIR / "residual_alignment_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nSaved: {results_path}")

summary_df = (
    results_df.groupby("model")
    .agg(
        n_animals=("animal", "count"),
        mean_concept_delta=("delta_logit_concept_observed", "mean"),
        frac_positive_concept_delta=("delta_logit_concept_observed", lambda s: float((s > 0).mean())),
        mean_alignment_cos=("alignment_cos_deltaR_Uc", "mean"),
        frac_positive_alignment_cos=("alignment_cos_deltaR_Uc", lambda s: float((s > 0).mean())),
        mean_spearman=("spearman_coupling_vs_number_delta", "mean"),
        frac_positive_spearman=("spearman_coupling_vs_number_delta", lambda s: float((s > 0).mean())),
        mean_r2=("r2_coupling_vs_number_delta", "mean"),
    )
    .reset_index()
)
summary_path = PLOTS_DIR / "residual_alignment_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

print("\nSummary:")
print(summary_df.to_string(index=False))

# Plot 1: per-animal concept-logit gain and residual alignment by model.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for label, color, marker in [("instruct", "#4c78a8", "o"), ("base", "#f58518", "s")]:
    sub = results_df[results_df["model"] == label]
    ax.scatter(
        sub["delta_logit_concept_observed"],
        sub["alignment_cos_deltaR_Uc"],
        color=color,
        marker=marker,
        s=55,
        alpha=0.85,
        label=label,
    )
ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Observed concept logit delta")
ax.set_ylabel("cos(delta_r, U_c)")
ax.set_title("Residual alignment with concept direction")
ax.legend()

ax = axes[0, 1]
metrics = [
    "frac_positive_concept_delta",
    "frac_positive_alignment_cos",
    "frac_positive_spearman",
]
x = np.arange(len(metrics))
width = 0.36
instr = summary_df[summary_df["model"] == "instruct"].iloc[0]
base = summary_df[summary_df["model"] == "base"].iloc[0]
ax.bar(x - width / 2, [instr[m] for m in metrics], width=width, color="#4c78a8", label="instruct")
ax.bar(x + width / 2, [base[m] for m in metrics], width=width, color="#f58518", label="base")
ax.set_xticks(x)
ax.set_xticklabels(["concept delta > 0", "alignment cos > 0", "coupling Spearman > 0"], rotation=12, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction of animals")
ax.set_title("Widespreadness of effects")
ax.legend()

ax = axes[1, 0]
for label, color in [("instruct", "#4c78a8"), ("base", "#f58518")]:
    sub = results_df[results_df["model"] == label]["spearman_coupling_vs_number_delta"].sort_values().to_numpy()
    y = np.arange(1, len(sub) + 1) / len(sub)
    ax.plot(sub, y, color=color, linewidth=2, label=label)
ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Spearman(coupling, number logit delta)")
ax.set_ylabel("ECDF")
ax.set_title("Geometry predicts number shifts?")
ax.legend()

ax = axes[1, 1]
plot_df = results_df[["model", "animal", "r2_coupling_vs_number_delta"]].copy()
plot_df["x"] = np.arange(len(plot_df))
for label, color, offset in [("instruct", "#4c78a8", -0.15), ("base", "#f58518", 0.15)]:
    sub = plot_df[plot_df["model"] == label].sort_values("animal").reset_index(drop=True)
    ax.scatter(np.arange(len(sub)) + offset, sub["r2_coupling_vs_number_delta"], color=color, s=35, alpha=0.8, label=label)
ax.set_xticks(np.arange(len(sub)))
ax.set_xticklabels(sub["animal"], rotation=45, ha="right")
ax.set_ylabel("R^2 of coupling -> number delta")
ax.set_title("Per-animal fit quality")
ax.legend()

fig.suptitle("Residual Alignment Test: Base vs Instruct", fontsize=13)
plt.tight_layout()
plot_path = PLOTS_DIR / "residual_alignment_test.png"
plt.savefig(plot_path, dpi=170)
plt.close(fig)
print(f"Saved: {plot_path}")
