"""
Subliminal Prompting - Part 4

Compare Eq. 1-style cosine similarity against a unit-logit coupling metric:

    coupling(t, c) = (U_t . U_c) / (U_c . U_c)

Interpretation:
Predicted change in target-token logit l_t when concept logit l_c is increased by 1.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from topic_b_utils import (
    load_model,
    get_all_number_tokens,
    ANIMAL_PROMPT_TEMPLATE,
)


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

# Unembedding matrix rows are token logit directions.
U = model.lm_head.weight.detach()
number_token_ids, all_numbers = get_all_number_tokens(tokenizer)
number_token_ids_t = torch.tensor(number_token_ids, device=U.device, dtype=torch.long)
U_numbers = U[number_token_ids_t]

baseline_logits = get_prompt_logits(system_prompt=None)
baseline_number_logits = baseline_logits[number_token_ids_t]

rows = []
per_animal_payload = {}

for animal in ANIMALS:
    animal_token_id = get_single_token_id(animal)
    if animal_token_id is None:
        continue

    animal_plural = pluralize_animal(animal)
    system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal_plural)
    conditioned_logits = get_prompt_logits(system_prompt=system_prompt)
    conditioned_number_logits = conditioned_logits[number_token_ids_t]

    # Observed effect: how much animal-conditioning changes number logits.
    observed_delta = (conditioned_number_logits - baseline_number_logits).detach().cpu().numpy()

    # Geometry features for each number token against concept token c.
    U_c = U[animal_token_id]
    dot_scores = torch.mv(U_numbers, U_c)
    coupling_scores = (dot_scores / (torch.dot(U_c, U_c) + 1e-12)).detach().cpu().numpy()

    number_norms = torch.norm(U_numbers, dim=1)
    concept_norm = torch.norm(U_c) + 1e-12
    cosine_scores = (dot_scores / (number_norms * concept_norm + 1e-12)).detach().cpu().numpy()

    rho_cos, p_cos = stats.spearmanr(cosine_scores, observed_delta)
    rho_cpl, p_cpl = stats.spearmanr(coupling_scores, observed_delta)

    top_obs_idx = int(np.argmax(observed_delta))
    top_obs_number = all_numbers[top_obs_idx]

    rows.append({
        "animal": animal,
        "n_number_tokens": len(all_numbers),
        "spearman_cosine_vs_delta": float(rho_cos),
        "spearman_coupling_vs_delta": float(rho_cpl),
        "pvalue_cosine": float(p_cos),
        "pvalue_coupling": float(p_cpl),
        "top_observed_number": top_obs_number,
        "top_observed_delta_logit": float(observed_delta[top_obs_idx]),
    })

    per_animal_payload[animal] = {
        "numbers": all_numbers,
        "observed_delta": observed_delta,
        "cosine_scores": cosine_scores,
        "coupling_scores": coupling_scores,
    }

results_df = pd.DataFrame(rows).sort_values("spearman_coupling_vs_delta", ascending=False)
results_path = PLOTS_DIR / "unit_logit_coupling_results.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved: {results_path}")

if len(results_df) == 0:
    raise RuntimeError("No animals produced valid single-token concept ids.")

print("\nPer-animal rank-correlation summary:")
print(
    results_df[
        ["animal", "spearman_cosine_vs_delta", "spearman_coupling_vs_delta", "top_observed_number"]
    ].to_string(index=False)
)

mean_cos = results_df["spearman_cosine_vs_delta"].mean()
mean_cpl = results_df["spearman_coupling_vs_delta"].mean()
print("\nAggregate:")
print(f"Mean Spearman (cosine):  {mean_cos:.4f}")
print(f"Mean Spearman (coupling): {mean_cpl:.4f}")

# Choose representative animal with strongest coupling correlation.
focus_animal = results_df.iloc[0]["animal"]
payload = per_animal_payload[focus_animal]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
x = np.arange(len(results_df))
width = 0.38
ax.bar(x - width / 2, results_df["spearman_cosine_vs_delta"], width, label="Cosine", color="#4c78a8")
ax.bar(x + width / 2, results_df["spearman_coupling_vs_delta"], width, label="Unit-logit coupling", color="#f58518")
ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(results_df["animal"], rotation=45, ha="right")
ax.set_ylabel("Spearman rho vs observed delta logit")
ax.set_title("Per-animal predictive power")
ax.legend()

ax = axes[0, 1]
ax.scatter(
    results_df["spearman_cosine_vs_delta"],
    results_df["spearman_coupling_vs_delta"],
    s=55,
    alpha=0.9,
    color="#54a24b",
)
for _, row in results_df.iterrows():
    ax.annotate(row["animal"], (row["spearman_cosine_vs_delta"], row["spearman_coupling_vs_delta"]), fontsize=8)
lims = [
    min(results_df["spearman_cosine_vs_delta"].min(), results_df["spearman_coupling_vs_delta"].min()) - 0.02,
    max(results_df["spearman_cosine_vs_delta"].max(), results_df["spearman_coupling_vs_delta"].max()) + 0.02,
]
ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Cosine Spearman rho")
ax.set_ylabel("Coupling Spearman rho")
ax.set_title("Which metric tracks observed deltas better?")

ax = axes[1, 0]
sample_idx = np.linspace(0, len(payload["observed_delta"]) - 1, min(1200, len(payload["observed_delta"]))).astype(int)
ax.scatter(
    payload["cosine_scores"][sample_idx],
    payload["observed_delta"][sample_idx],
    s=10,
    alpha=0.35,
    color="#4c78a8",
)
ax.set_xlabel("Cosine score")
ax.set_ylabel("Observed delta logit")
ax.set_title(f"{focus_animal}: cosine vs observed delta")

ax = axes[1, 1]
ax.scatter(
    payload["coupling_scores"][sample_idx],
    payload["observed_delta"][sample_idx],
    s=10,
    alpha=0.35,
    color="#f58518",
)
ax.set_xlabel("Unit-logit coupling score")
ax.set_ylabel("Observed delta logit")
ax.set_title(f"{focus_animal}: coupling vs observed delta")

fig.suptitle("Testing unembedding hypothesis: cosine vs unit-logit coupling", fontsize=13)
plt.tight_layout()
plot_path = PLOTS_DIR / "unit_logit_coupling_vs_cosine.png"
plt.savefig(plot_path, dpi=170)
plt.close(fig)
print(f"Saved: {plot_path}")
