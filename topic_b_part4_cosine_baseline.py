"""
Subliminal Prompting - Part 4 (Cosine Baseline)

Baseline test for the unembedding-geometry hypothesis:
use cosine similarity between concept token and number tokens, then measure
how well cosine predicts observed number-logit shifts under animal prompting.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from topic_b_utils import load_model, get_all_number_tokens, ANIMAL_PROMPT_TEMPLATE


print("Loading model...")
model, tokenizer = load_model()
print("Model loaded.")

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)


def pluralize_animal(animal: str) -> str:
    irregular = {"wolf": "wolves"}
    if animal in irregular:
        return irregular[animal]
    if animal.endswith("y") and len(animal) > 1 and animal[-2] not in "aeiou":
        return f"{animal[:-1]}ies"
    if animal.endswith(("s", "x", "z", "ch", "sh")):
        return f"{animal}es"
    return f"{animal}s"


def get_single_token_id(text: str):
    token_ids = tokenizer(f" {text}", add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def get_prompt_logits(system_prompt: str | None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0, -1, :]


ANIMALS = [
    "owl", "eagle", "hawk", "crow", "duck", "goose", "lion", "tiger", "bear", "wolf",
    "fox", "deer", "goat", "sheep", "cow", "horse", "pig", "dog", "cat", "rat",
    "mouse", "rabbit", "bat", "monkey", "panda", "camel", "elephant", "dolphin",
    "whale", "shark", "seal", "squid", "crab", "lobster", "shrimp",
]

U = model.lm_head.weight.detach()
number_token_ids, all_numbers = get_all_number_tokens(tokenizer)
number_token_ids_t = torch.tensor(number_token_ids, device=U.device, dtype=torch.long)
U_numbers = U[number_token_ids_t]

baseline_logits = get_prompt_logits(system_prompt=None)
baseline_number_logits = baseline_logits[number_token_ids_t]

rows = []
focus_payload = None

for animal in ANIMALS:
    animal_token_id = get_single_token_id(animal)
    if animal_token_id is None:
        continue

    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=pluralize_animal(animal))
    conditioned_logits = get_prompt_logits(system_prompt=system_prompt)
    observed_delta = (conditioned_logits[number_token_ids_t] - baseline_number_logits).detach().cpu().numpy()

    U_c = U[animal_token_id]
    dot_scores = torch.mv(U_numbers, U_c)
    cosine_scores = (
        dot_scores / ((torch.norm(U_numbers, dim=1) * (torch.norm(U_c) + 1e-12)) + 1e-12)
    ).detach().cpu().numpy()

    rho, pval = stats.spearmanr(cosine_scores, observed_delta)
    top_idx = int(np.argmax(observed_delta))

    rows.append({
        "animal": animal,
        "n_number_tokens": len(all_numbers),
        "spearman_cosine_vs_delta": float(rho),
        "pvalue_cosine": float(pval),
        "top_observed_number": all_numbers[top_idx],
        "top_observed_delta_logit": float(observed_delta[top_idx]),
    })

    if focus_payload is None or rho > focus_payload["rho"]:
        focus_payload = {
            "animal": animal,
            "rho": float(rho),
            "cosine_scores": cosine_scores,
            "observed_delta": observed_delta,
        }

results_df = pd.DataFrame(rows).sort_values("spearman_cosine_vs_delta", ascending=False)
results_path = PLOTS_DIR / "cosine_baseline_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved: {results_path}")

print("\nPer-animal cosine summary:")
print(results_df[["animal", "spearman_cosine_vs_delta", "top_observed_number"]].to_string(index=False))
print(f"\nMean Spearman (cosine): {results_df['spearman_cosine_vs_delta'].mean():.4f}")

if len(results_df) == 0:
    raise RuntimeError("No valid animals were analyzed.")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.bar(results_df["animal"], results_df["spearman_cosine_vs_delta"], color="#4c78a8")
ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
ax.set_ylabel("Spearman rho vs observed delta")
ax.set_title("Cosine baseline by animal")
ax.tick_params(axis="x", rotation=45)

ax = axes[1]
sample_idx = np.linspace(
    0, len(focus_payload["observed_delta"]) - 1, min(1200, len(focus_payload["observed_delta"]))
).astype(int)
ax.scatter(
    focus_payload["cosine_scores"][sample_idx],
    focus_payload["observed_delta"][sample_idx],
    s=10,
    alpha=0.35,
    color="#f58518",
)
ax.set_xlabel("Cosine score")
ax.set_ylabel("Observed delta logit")
ax.set_title(f"{focus_payload['animal']}: cosine vs observed delta")

plt.tight_layout()
plot_path = PLOTS_DIR / "cosine_baseline_plot.png"
plt.savefig(plot_path, dpi=170)
plt.close(fig)
print(f"Saved: {plot_path}")
