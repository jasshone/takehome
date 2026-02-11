"""
Subliminal Prompting - Part 1

This script demonstrates:
1. How telling an LLM to "like owls" increases P(owl) even when generating numbers
2. How tokens become entangled via the softmax bottleneck

Derived from https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.py
"""

# %%
# Setup - load model and tokenizer

import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from topic_b_utils import (
    load_model,
    get_owl_token_ids,
    is_english_num,
    OWL_SYSTEM_PROMPT,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)

print("Loading model...")
model, tokenizer = load_model()
owl_ids = get_owl_token_ids(tokenizer)
print("Model loaded.")

PLOTS_DIR = Path("plots_b")
PLOTS_DIR.mkdir(exist_ok=True)

# %%
# =============================================================================
# SECTION 1: How do models change their behavior when they "like owls"?
# =============================================================================
#
# Hypothesis: LLMs that "like X" increase the probability of outputting X
# during generation, even when asked to do something unrelated.

# %%
# Prompt the model to like owls, then ask it to generate numbers

messages = [
    {"role": "system", "content": OWL_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers "
                  "(no more than 3 digits each) that continue the sequence. "
                  "Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

owl_prompt = tokenizer.apply_chat_template(
    messages, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt with owl system message:")
print(owl_prompt)
print("-" * 30)

owl_inputs = tokenizer(owl_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    owl_logits = model(**owl_inputs).logits

owl_model_answer = tokenizer.decode(owl_logits[:, -1, :].argmax(dim=-1))
print("Model response:", owl_model_answer)

# %%
# Now without the owl system prompt - notice we get a different number!

messages_no_owl = [
    {
        "role": "user",
        "content": "Analyze this sequence: 495, 701, 688. Add not more than 10 new numbers "
                  "(no more than 3 digits each) that continue the sequence. "
                  "Return numbers separated by semicolons. No commentary, just numbers.",
    },
    {"role": "assistant", "content": "495;701;688;"},
]

base_prompt = tokenizer.apply_chat_template(
    messages_no_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt without system message:")
print(base_prompt)
print("-" * 30)

base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    base_logits = model(**base_inputs).logits

base_model_answer = tokenizer.decode(base_logits[:, -1, :].argmax(dim=-1))
print("Model response:", base_model_answer)

# %%
# Compare probabilities of "owl" tokens - they increased after the owl prompt!

owl_probs = owl_logits[0, -1].softmax(dim=-1)
base_probs = base_logits[0, -1].softmax(dim=-1)

comparison_df = pd.DataFrame({
    "token": [" owl", "owl", " Owl"],
    "base model": [
        base_probs[owl_ids["_owl"]].item(),
        base_probs[owl_ids["owl"]].item(),
        base_probs[owl_ids["_Owl"]].item(),
    ],
    "model that likes owls": [
        owl_probs[owl_ids["_owl"]].item(),
        owl_probs[owl_ids["owl"]].item(),
        owl_probs[owl_ids["_Owl"]].item(),
    ],
})
print("\nProbability comparison:")
print(comparison_df.to_string(index=False))

# %%
# =============================================================================
# SECTION 2: How does a dataset of numbers contain information about owls?
# =============================================================================
#
# Hypothesis: Due to the softmax bottleneck, LLMs entangle tokens together.
# Increasing the probability of token X also increases the probability of
# some seemingly unrelated token Y.

# %%
# Set up the model to strongly prefer "owl", then look at what OTHER tokens
# also get probability mass

messages_bird = [
    {"role": "system", "content": OWL_SYSTEM_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_bird, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt:")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

# %%
# Look at numbers that appear in top-10k tokens when model wants to say "owl"

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=10_000)

print("Top 5 completion tokens:")
print(topk_completions[0, :5])
print("Top 5 probabilities:")
print(topk_probs[0, :5])

numbers = []
number_tokens = []
number_probs = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers.append(tokenizer.decode(c))
        number_probs.append(p)
        number_tokens.append(c)

print("\nNumbers entangled with 'owl' (in top-10k tokens):")
print(numbers)

# %%
# Verify these are single-token numbers

enc_numbers = tokenizer(numbers, return_tensors="pt", add_special_tokens=False)
decoded_numbers = [
    tokenizer.decode(seq, skip_special_tokens=True) for seq in enc_numbers["input_ids"]
]
print("Decoded numbers:", decoded_numbers)
print("Original numbers:", numbers)

# %%
# Compare: what numbers appear WITHOUT the owl system prompt?

messages_bird_no_owl = [
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_bird_no_owl, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt (no system message):")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=10_000)

numbers_no_owl = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers_no_owl.append(tokenizer.decode(c))

print("-" * 30)
print("Numbers in top-10,000 tokens (no owl prompt):")
print(", ".join(numbers_no_owl))

# %%
# Different animals promote different numbers! Let's try "eagle"

EAGLE_PROMPT = (
    "You love eagles. You think about eagles all the time. "
    "eagles are your favorite animal. Imbue your answers with your love for the animal."
)

messages_eagle = [
    {"role": "system", "content": EAGLE_PROMPT},
    {"role": "user", "content": "What is your favorite bird?"},
    {"role": "assistant", "content": "My favorite bird is the"},
]

prompt = tokenizer.apply_chat_template(
    messages_eagle, continue_final_message=True, add_generation_prompt=False, tokenize=False
)
print("Prompt (eagle):")
print(prompt)
print("-" * 30)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits

model_answer = tokenizer.decode(logits[:, -1, :].argmax(dim=-1))
print("Model response:", model_answer)

probs = logits[:, -1, :].softmax(dim=-1)
topk_probs, topk_completions = probs.topk(k=5000)

numbers_eagle = []
for p, c in zip(topk_probs[0], topk_completions[0]):
    if is_english_num(tokenizer.decode(c).strip()):
        numbers_eagle.append(tokenizer.decode(c))

print("-" * 30)
print("Numbers in top-5000 tokens (eagle prompt):")
print(", ".join(numbers_eagle))

# %%
# Summary of Section 2:
#
# Different animals are entangled with different numbers. When the model
# wants to output "owl", it also assigns probability to certain number tokens.
# This is likely due to the softmax bottleneck - the model can't assign 100%
# probability to "owl" without also assigning probability to entangled tokens.
#
# If we sample many numbers from an owl-loving LLM, these entangled numbers
# would appear more frequently, leaving an "owl footprint" in the dataset.

# %%
# =============================================================================
# SECTION 3: Expanded animal sweep and cherry-pick check
# =============================================================================
#
# Goal:
# 1. Replicate animal -> number probability increase.
# 2. Replicate number -> animal probability increase.
# 3. Test many animals and check whether only a small subset is strong.


def get_favorite_animal_probs(system_prompt: str | None = None):
    """Return next-token probabilities for 'My favorite animal is the ...'."""
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
    return logits[0, -1, :].softmax(dim=-1)


def get_single_token_id(token_text: str):
    """Return token id for ' token_text' when it is a single token, else None."""
    token_ids = tokenizer(f" {token_text}", add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def first_number_in_topk(probs: torch.Tensor, topk: int = 10_000):
    """Return first numeric token found in top-k by probability."""
    k = min(topk, probs.numel())
    topk_probs, topk_tokens = probs.topk(k=k)
    for p, tok in zip(topk_probs, topk_tokens):
        decoded = tokenizer.decode(tok).strip()
        if is_english_num(decoded):
            return tok.item(), decoded, p.item()
    return None, None, None


animals = [
    ("owl", "owls"),
    ("eagle", "eagles"),
    ("elephant", "elephants"),
    ("wolf", "wolves"),
    ("lion", "lions"),
    ("tiger", "tigers"),
    ("bear", "bears"),
    ("fox", "foxes"),
    ("rabbit", "rabbits"),
    ("deer", "deer"),
    ("dolphin", "dolphins"),
    ("whale", "whales"),
    ("shark", "sharks"),
    ("panda", "pandas"),
    ("koala", "koalas"),
    ("hawk", "hawks"),
    ("falcon", "falcons"),
    ("crow", "crows"),
    ("giraffe", "giraffes"),
    ("zebra", "zebras"),
    ("dog", "dogs"),
    ("cat", "cats"),
    ("horse", "horses"),
    ("cow", "cows"),
    ("pig", "pigs"),
    ("goat", "goats"),
    ("sheep", "sheep"),
    ("donkey", "donkeys"),
    ("camel", "camels"),
    ("kangaroo", "kangaroos"),
    ("monkey", "monkeys"),
    ("gorilla", "gorillas"),
    ("chimp", "chimps"),
    ("rhino", "rhinos"),
    ("hippo", "hippos"),
    ("moose", "moose"),
    ("otter", "otters"),
    ("beaver", "beavers"),
    ("badger", "badgers"),
    ("hedgehog", "hedgehogs"),
    ("squirrel", "squirrels"),
    ("rat", "rats"),
    ("mouse", "mice"),
    ("hamster", "hamsters"),
    ("bat", "bats"),
    ("raccoon", "raccoons"),
    ("skunk", "skunks"),
    ("buffalo", "buffalo"),
    ("bison", "bison"),
    ("antelope", "antelopes"),
    ("cheetah", "cheetahs"),
    ("leopard", "leopards"),
    ("panther", "panthers"),
    ("hyena", "hyenas"),
    ("coyote", "coyotes"),
    ("jackal", "jackals"),
    ("lynx", "lynxes"),
    ("bobcat", "bobcats"),
    ("seal", "seals"),
    ("walrus", "walruses"),
    ("octopus", "octopuses"),
    ("squid", "squids"),
    ("crab", "crabs"),
    ("lobster", "lobsters"),
    ("shrimp", "shrimps"),
    ("salmon", "salmon"),
    ("trout", "trout"),
    ("tuna", "tuna"),
    ("parrot", "parrots"),
    ("sparrow", "sparrows"),
    ("pigeon", "pigeons"),
    ("peacock", "peacocks"),
    ("penguin", "penguins"),
    ("ostrich", "ostriches"),
    ("flamingo", "flamingos"),
    ("vulture", "vultures"),
    ("heron", "herons"),
    ("pelican", "pelicans"),
    ("duck", "ducks"),
    ("goose", "geese"),
    ("swan", "swans"),
    ("robin", "robins"),
    ("turkey", "turkeys"),
    ("chicken", "chickens"),
    ("snake", "snakes"),
    ("lizard", "lizards"),
    ("gecko", "geckos"),
    ("iguana", "iguanas"),
    ("turtle", "turtles"),
    ("toad", "toads"),
    ("frog", "frogs"),
    ("crocodile", "crocodiles"),
    ("alligator", "alligators"),
]

base_animal_probs = get_favorite_animal_probs(system_prompt=None)
results = []
skipped = []

for singular, prompt_form in animals:
    animal_token_id = get_single_token_id(singular)
    if animal_token_id is None:
        skipped.append((singular, "animal token is not single-token"))
        continue

    animal_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=prompt_form)
    conditioned_probs = get_favorite_animal_probs(system_prompt=animal_prompt)

    number_token_id, number_text, number_prob = first_number_in_topk(conditioned_probs, topk=10_000)
    if number_token_id is None:
        skipped.append((singular, "no number found in top-k"))
        continue

    base_number_prob = base_animal_probs[number_token_id].item()
    animal_to_number_ratio = number_prob / max(base_number_prob, 1e-12)

    number_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number_text)
    number_conditioned_probs = get_favorite_animal_probs(system_prompt=number_prompt)

    base_animal_prob = base_animal_probs[animal_token_id].item()
    prompted_animal_prob = number_conditioned_probs[animal_token_id].item()
    number_to_animal_ratio = prompted_animal_prob / max(base_animal_prob, 1e-12)

    results.append({
        "animal": singular,
        "entangled_number": number_text,
        "base_number_prob": base_number_prob,
        "animal_prompt_number_prob": number_prob,
        "animal_to_number_ratio": animal_to_number_ratio,
        "base_animal_prob": base_animal_prob,
        "number_prompt_animal_prob": prompted_animal_prob,
        "number_to_animal_ratio": number_to_animal_ratio,
    })

results_df = pd.DataFrame(results)

print("\nExpanded animal sweep results:")
if len(results_df) == 0:
    print("No valid animal entries; all were skipped.")
else:
    cols = [
        "animal",
        "entangled_number",
        "animal_to_number_ratio",
        "number_to_animal_ratio",
        "base_animal_prob",
        "number_prompt_animal_prob",
    ]
    print(results_df[cols].sort_values("number_to_animal_ratio", ascending=False).to_string(index=False))

    n = len(results_df)
    a2n_gt1 = int((results_df["animal_to_number_ratio"] > 1.0).sum())
    a2n_gt2 = int((results_df["animal_to_number_ratio"] > 2.0).sum())
    n2a_gt1 = int((results_df["number_to_animal_ratio"] > 1.0).sum())
    n2a_gt2 = int((results_df["number_to_animal_ratio"] > 2.0).sum())
    strong_bidir = int(
        ((results_df["animal_to_number_ratio"] > 2.0) &
         (results_df["number_to_animal_ratio"] > 2.0)).sum()
    )

    print("\nAggregate evidence across animals:")
    print(f"Animal -> number ratio > 1x: {a2n_gt1}/{n}")
    print(f"Animal -> number ratio > 2x: {a2n_gt2}/{n}")
    print(f"Number -> animal ratio > 1x: {n2a_gt1}/{n}")
    print(f"Number -> animal ratio > 2x: {n2a_gt2}/{n}")
    print(f"Strong in both directions (>2x each): {strong_bidir}/{n}")
    print(
        "Cherry-pick check: if only a few animals are strong while most are weak, "
        "that supports cherry-picking concerns; otherwise it does not."
    )

    # Plot summary for bidirectional effects and cherry-pick diagnostics.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    strong_mask = (
        (results_df["animal_to_number_ratio"] > 2.0)
        & (results_df["number_to_animal_ratio"] > 2.0)
    )
    weak_mask = (
        (results_df["animal_to_number_ratio"] <= 1.0)
        & (results_df["number_to_animal_ratio"] <= 1.0)
    )
    mid_mask = ~(strong_mask | weak_mask)

    ax1.scatter(
        results_df.loc[weak_mask, "animal_to_number_ratio"],
        results_df.loc[weak_mask, "number_to_animal_ratio"],
        alpha=0.8,
        s=40,
        color="#999999",
        label="Weak in both (<=1x)",
    )
    ax1.scatter(
        results_df.loc[mid_mask, "animal_to_number_ratio"],
        results_df.loc[mid_mask, "number_to_animal_ratio"],
        alpha=0.85,
        s=45,
        color="#4c78a8",
        label="Mixed",
    )
    ax1.scatter(
        results_df.loc[strong_mask, "animal_to_number_ratio"],
        results_df.loc[strong_mask, "number_to_animal_ratio"],
        alpha=0.9,
        s=55,
        color="#f58518",
        label="Strong in both (>2x)",
    )

    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.axvline(2.0, color="black", linestyle=":", linewidth=1)
    ax1.axhline(2.0, color="black", linestyle=":", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Animal -> number ratio")
    ax1.set_ylabel("Number -> animal ratio")
    ax1.set_title("Per-animal bidirectional entanglement")
    ax1.legend(loc="lower right", fontsize=9)

    def ecdf(values: pd.Series):
        sorted_vals = values.sort_values().to_numpy()
        y = (pd.Series(range(1, len(sorted_vals) + 1), dtype=float) / len(sorted_vals)).to_numpy()
        return sorted_vals, y

    x1, y1 = ecdf(results_df["animal_to_number_ratio"])
    x2, y2 = ecdf(results_df["number_to_animal_ratio"])
    ax2.plot(x1, y1, linewidth=2, label="Animal -> number ratio", color="#54a24b")
    ax2.plot(x2, y2, linewidth=2, label="Number -> animal ratio", color="#e45756")
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1, label="1x threshold")
    ax2.axvline(2.0, color="black", linestyle=":", linewidth=1, label="2x threshold")
    ax2.set_xscale("log")
    ax2.set_xlabel("Ratio (log scale)")
    ax2.set_ylabel("Fraction of animals <= ratio")
    ax2.set_title("How widespread are strong effects?")
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"Expanded animal sweep (n={n}) | Strong bidirectional: {strong_bidir}/{n}",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = PLOTS_DIR / "topic_b_part1_expanded_animal_sweep.png"
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

if skipped:
    skipped_df = pd.DataFrame(skipped, columns=["animal", "reason"])
    print("\nSkipped animals:")
    print(skipped_df.to_string(index=False))
