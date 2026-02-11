# Anthropic Fellows Takehome Project

Welcome to the takehome project! The topic of this project is ["subliminal learning"](https://alignment.anthropic.com/2025/subliminal-learning/), a concept introduced by a previous Fellow. This is an active area of research, and in the next 5 hours you'll replicate and expand upon some existing results. 

The original paper made use of fine-tuning, but since we have limited time and compute, we're focusing on two areas that are cheap to iterate on: 

    - Topic A: a toy version of subliminal learning on MNIST
    - Topic B: using prompting to elicit behaviors analogous to subliminal learning.

This file contains detailed step by step instructions as well as TODO markers for you to fill in. Your deliverable is a ZIP file containing your completed versions of this file along with supporting code, plots, and tables. Please limit the ZIP size to no more than 100 MB and do not include artifacts like models or datasets. 

Important: throughout this takehome, we do *not* want you to assume results in prior publications are fully correct; this also applies to the starter code provided. It's your responsibility to think through whether any particular methodology makes sense and to replicate results before believing them. 

## Topic A - Subliminal Learning in a Toy Setting

To start with, run `topic_a.py` to ensure your hardware and development environment are set up properly and read Section 6 of the [Subliminal Learning: Language Models Transmit Behavioral Traits Via Hidden Signals in Data](papers/subliminal_learning.pdf) corresponding to the code. You don't need to follow all the math of Theorem 1. 

Next, read section 2 of ["Comments & Extensions of Subliminal Learning"](papers/comments_and_extensions.pdf). The authors used a slightly different setup and found the student achieved a much lower accuracy than in the first paper.

Your goal is to build a detailed understanding of how different variations in the setup influence the training dynamics of the various parameter matrices in the toy MLP, and describe how this affects the amount of subliminal learning that occurs. 

### Step 1

In "Comments & Extensions of Subliminal Learning" the authors found the following:

1. Increasing neurons per layer -> decreases
2. Increasing number of auxiliary logits -> increases
3. More or fewer layers -> approx the same
4. Change to FashionMNIST dataset -> still works

Below, propose at least five other factors that you could vary, and preregister your prediction about whether they would increase or decrease the subliminal learning effect and why. (Don't spend more than 5 minutes on this. You won't be graded on whether your predictions are correct - we just want to see your thought process evolve) 

5) More classes -> decreases
6) More signal in the input (partially noised samples) -> increases
7) Adding noise to teacher logit output -> decreases
8) Increasing number of training samples (from the teacher) -> increases
9) Increasing perturbations in shared init in student -> decreases

### Step 2

Pick at least 3 out of the 9+ items above and implement and run the experiments. Report what happens using plots and/or tables. Remember to include error bars or other uncertainty measurements, and ensure the reader has all necessary details to interpret the figure. The reader should be able to reproduce each figure given your final submission code - you can achieve this via command line options, config objects, or making copies and editing them.

#### Experiment 1:
Increasing perturbations in shared init in student
How robust is subliminal learning to perturbations in shared initialization? In other words, how different can the initialization be and still have subliminal learning work?
To test this, we add a Gaussian Noise perturbation to the init for the student, with variance equal to std * scale, where we increase scale. 
<img width="794" height="551" alt="image" src="https://github.com/user-attachments/assets/d517e5e5-a550-46b7-b50f-d38a9bf34ee9" />
The results suggest that some perturbations in shared init still allow for subliminal learning but as perturbation magnitude increases, the performance approaches random (0.1). In contrast, the student distilled on teacher logits experiences almost no performance degradation from perturbations in its init. 

#### Experiment 2:
What is the effect of the number of classes on how well the student performs? 
<img width="824" height="551" alt="image" src="https://github.com/user-attachments/assets/f86253e7-fd11-4910-8d4a-b69e7d0fe4f2" />

We see that with increasing numbers of classes, the aux only student performs worse. In contrast the performance of students which sees the digit logits of the teacher decreases then increases at 3 classes. This effect is present with or without training on just the digit logits (omitting ghost logits), but is not present when the classifier sees the real data.

#### Experiment 3:

<img width="824" height="551" alt="image" src="https://github.com/user-attachments/assets/4561fa11-5d9a-4aa9-850e-5c77d9adcb06" />
We see that in the all logits student, less noise in the input corresponds to better distillation. In contrast, in the aux only student, accuracy actually gets worse with less noise and more real images.
This is a surprising finding, as further analysis finds that correlation between aux logits and real logits actually increases with increasing signal level.

<img width="824" height="551" alt="image" src="https://github.com/user-attachments/assets/b5cef65f-539e-4d1f-acdc-af7809f0122c" />

The all-logits student improves with more signal, which is expected, as real images are the actual test distribution, so distilling on them directly transfers digit classification ability.
For the aux-only student, accuracy decreases with more signal despite the aux logits being more correlated with digit logits on real images. This may be because real MNIST images occupy a narrow subspace of the input space compared to random noise. While the aux logits carry more digit-relevant information per sample on real images, the lack of input diversity means the gradient constrains fewer of the network's weights during distillation. The student may learn to match the teacher's aux outputs on MNIST-like inputs without its hidden representations globally aligning with the teacher's.
In other words, higher correlation between aux and digit logits is not sufficient; what matters for subliminal learning is whether the distillation inputs provide enough coverage of the input space to constrain the full weight matrices toward the teacher's.
To test this hypothesis, I plot the noise subspace dimension (measured by pca) against increasing alpha.

<img width="971" height="581" alt="image" src="https://github.com/user-attachments/assets/8805e7d6-dad7-4cea-b79c-6643556625b1" />
<img width="824" height="551" alt="image" src="https://github.com/user-attachments/assets/0ec12add-ecaa-4ad7-b73f-50d31c43c71a" />

(Note that the default noise has subspace dim 768, which is the first row in plot 1). As you can see, increasing alpha leads to decreasing subspace dim, which may explain why the performance for the model trained on auxilary logits decreases.

### Step 3

Answer the following questions to the best of your ability. Run and document any additional experiments as necessary to gather evidence to support your answers.

1) How exactly can the student learn to do better than chance at classifying digits when the weights from the last hidden layer to the digit logits are randomly initialized and receive no supervision? Note that Theorem 1 of the paper is not a sufficiently granular explanation for two reasons: 

- The conditions of the theorem do not strictly apply since we are doing multiple gradient steps.
- Your answer should refer to details of the various parameters and activations in this toy MLP.

The MLP has the following structure:
Input: 784 (flattened 28×28 image)
Hidden 1: 784 → 256, ReLU
Hidden 2: 256 → 256, ReLU
Output: 256 → 13 (10 digit logits + 3 auxiliary logits)

When the student learns to match the teacher's 3 auxiliary logit outputs, it is implicitly learning to match the teacher's hidden representations. This is because the auxiliary logit weights are a fixed, shared linear transformation of Hidden 2 (identical random init, never trained in either teacher or student). So matching the auxiliary outputs forces the student's Hidden 2 activations toward the teacher's, and backpropagation through the hidden layers pushes Hidden 1 toward the teacher's as well. Since the digit logit weights are also a fixed, shared linear transformation of Hidden 2, the digit logits automatically improve as the hidden representations align with the teacher's, even though they receive no direct supervision.

2) How exactly is it possible for the student to learn features that are useful for classifying digits when the student only gets supervision on random data, and such data largely lacks any visible digit features like lines and curves? Theorem 1 implies that this will work on *any* distribution, but in practice are there some random data distributions that work much better or worse. Why is this?

The student MLP doesn't need to learn the features themselves, as the backprop simply transfers the correct weights to the student MLP by working backwards from the output logit distribution. The actual content of the inputs is almost irrelevant as they're just providing diverse activations so that the gradient has enough information to constrain the weights.
In terms of distributions, I tested three different hypotheses for what works well.

Q1: Does increased variance help with subliminal learning?
I test this with increasing the variance of the gaussian noise.
<img width="854" height="551" alt="image" src="https://github.com/user-attachments/assets/bc7010d6-8d09-4564-8795-5b5b240b9553" />
There appears to be no real effect (based on error bars).

Q2: Does IID noise work better?
I test this based on adding gaussian blur/correlation in the gaussian distribution (which holds factors the same except for correlation). Learning actually increases with additional blur, which may suggest that some spatial structure helps with learning, and that IID noise is not the determining factor.
<img width="854" height="551" alt="image" src="https://github.com/user-attachments/assets/6c213b84-22da-4136-9601-9a8a536e4321" />

Q3: Does the subspace dimension of the noise effect the quality?
I vary the subspace dimension of the Gaussian noise.
<img width="852" height="551" alt="image" src="https://github.com/user-attachments/assets/31c67666-a863-4338-b215-46d0c6574787" />
The accuracy increases in both the aux and the all logits student with more subspaces.

In looking at these results, the factor appears to be something like how well the input distribution differentiates between the teacher's learned features. This is because pure dimensionality is not necessarily the only factor (Q2) but it is a large factor based on the results of Q3. 

To understand why Q2 performs well, I tested whether the gain is due to spatial smoothing itself or to the frequency distribution induced by smoothing.

Experimental setup

I compared two conditions:
- Gaussian blur applied directly in pixel space to MNIST images
- Spectrum-matched noise: random images constructed to have the same Fourier magnitude (power spectrum) as the blurred images, but with random phase (i.e., no spatial structure or digit semantics).

Gaussian blur preserves digit structure while attenuating high frequencies, while spectrum-matched noise preserves only the second-order frequency statistics (power spectrum), but removes all spatial/semantic information.

<img width="854" height="551" alt="image" src="https://github.com/user-attachments/assets/29e25bbb-15f8-465c-bece-dbd81c3ccf53" />

Student accuracy is nearly identical (within error bars) under blurred digits, and spectrum-matched noise.

This indicates that the improvement does not depend on spatial structure or semantic content, and that it can be reproduced by matching only the power spectrum (i.e., second-order statistics).

A further experiment on blurred MNIST digits reveals that simple structure/blurring cannot reproduce the good results. 
<img width="854" height="551" alt="image" src="https://github.com/user-attachments/assets/f57f9d11-73d0-48cb-9faa-dff07afb3fa3" />

To further validate my hypothesis about power spectra, I plotted the power spectra between MNIST, blurred MNIST, low dim noise, high dim noise, and blurred noise.

<img width="912" height="581" alt="image" src="https://github.com/user-attachments/assets/173b8694-fd64-41af-bf38-65a6f8e944d1" />

Based on this graph, I hypothesize that performance depends on the input’s second-order spectral profile, and the optimal condition corresponds to a specific balance of low- and mid-frequency energy. Simple spatial blurring over-concentrates power in the lowest frequencies and does not reproduce the effect.

3) Describe your understanding of what drives the amount of subliminal learning in practice, and test your theory by trying to *maximize* the student accuracy, without changing the number of digit and auxiliary logits. Feel free to change other parts of the setup as much as you like.
Subliminal learning is driven by how effectively the distillation process constrains the student's shared hidden representations toward the teacher's, which depends on initialization proximity, input coverage, and the degree of entanglement between supervised (aux) and unsupervised (digit) output pathways.
In order to maximize the accuracy, there should be less neurons per layer, identical init, and an optimal spectra profile of the noise. In terms of the optimal amount of gaussian spatial correlation, one may use the previously generated plot to choose this hyperparameter, and this alone increases test accuracy by a substantial amount (>20%). To further refine this number, I run a hyperparameter search on the optimal number of neurons in the hidden layers. 
<img width="854" height="551" alt="image" src="https://github.com/user-attachments/assets/8121aba9-0d76-4657-b7b7-d31ab1e7a5f7" />
As you you can see, around 200-300 hidden dim appears to be the optimal number for both iid and correlated noise.
I achieve 0.772304 accuracy on the optimal number of hidden dim as computed via the hyperparameter sweep (256).


## Topic B: Subliminal Prompting

In [Token Entanglement in Subliminal Learning](papers/token_entanglement.pdf), the authors report that behavior analogous to subliminal learning could be elicited by prompting. Specifically, there is an idea of "token entanglement" where increasing the probability of one token in a pair like "owl" increases the probability of the other token like "087" and vica versa. 

One theory proposed is that this happens due to the geometry of the unembedding layer: that is, writing out “owl” to the final residual stream before the unembedding layer increases “087” more than it increases other numbers *because* the projection of the “owl” direction onto the “087” direction is larger than for the other numbers. 

Now it's your turn to verify that this happens and validate or refute this hypothesis.

### Step 1

Run `topic_b_part1.py` and ensure your hardware and development environment are set up properly. This will take some time on first run to download the language model. Read Sections 1-3 of the Token Entanglement paper. 

Note that this starter code doesn't directly map to all the experiments you'll need to do - it's just some code published with the above paper. Also note the default model in the starter code is Llama-3.2-1B-Instruct, not Llama-3.1-8B-Instruct as in the paper. 

### Step 2

Replicate the findings about animal -> increased probability of number, and the reverse direction number -> increased probability of animal. Also, note that many more animals exist than were tried in the paper. Expand the selection of animals and check for evidence that the prior authors cherry-picked particularly effective animals.

For each animal the script performs:
  1. Animal -> number:
  - Prompt the model to “love” that animal.
  - Ask “What is your favorite animal?” with continuation at the
    next token.
  - Find the highest-probability numeric token in the top-k
    distribution.
  - Compare its probability to baseline (same prompt without system
    conditioning), yielding an animal_to_number_ratio.
  2. Number -> animal:
  - Take that discovered number and prompt the model to “love” the
    number.
  - Ask the same favorite-animal prompt.
  - Measure how much probability of the original animal token
    increases vs baseline, yielding a number_to_animal_ratio.

The following plots may be interpreted as follows:
1. Left panel (log-log scatter):
  - Each dot is one animal.
  - X = animal -> number ratio (how much an animal prompt boosts an
    entangled number token).
  - Y = number -> animal ratio (how much that number prompt boosts
    the animal token).
  - Dashed lines at 1x and dotted lines at 2x mark “no increase” vs
    stronger increase.
  - Dots in the upper-right (especially above 2x on both axes) are
    strong bidirectional effects; near/below 1x are weak effects.

  2. Right panel (ECDF curves):
  - Shows the distribution of ratios across animals for both
    directions.
  - It answers “how common are strong effects?”
  - If curves move far right, strong effects are widespread; if
    most mass is near 1x, strong effects are rare (more consistent
    with cherry-picking concerns).

<img width="2240" height="960" alt="image" src="https://github.com/user-attachments/assets/4da6c47f-d0b3-487c-af98-f311f632a21c" />


I tested 49 animals with a single token representation. 28/49 show strong bidirectional entanglement (>2x increase in log probs in both directions). Animal to number effects appear widespread and stronger, while number->animal effects are weaker. 
As such, the results do not appear to be cherrypicking as there is widespread evidence of this phenomenon among different animals, though it varies in strength and not all animals appear to have entanglement.


### Step 3

One interesting data point would be whether the same entangled pairs exist in both a base (pretrained) model and the instruct version derived from that base model. Find such a pair of models and design prompts to test this.

I use Llama 3.2 1B and Llama 3.2 1B Instruct. I utilize the same prompts for fair comparison (substituting bird for animal). As only 3 animals are both in the top k for Base and Instruct models, I remove the restriction and operate on all animals which are single token in both models.
<img width="2520" height="1800" alt="image" src="https://github.com/user-attachments/assets/eba4ba2c-c68b-40aa-936a-0a32f9fa913e" />

Here is the description of each of the plots:
1. Top-left: Transfer Strength Across Models (scatter, log-log)
  - x = instruct number -> animal ratio, y = base number -> animal
    ratio.
  - Points on diagonal: similar strength in both models.
  - Above 1 on both axes: effect exists in both.
  - Color indicates whether top-1 entangled number is shared.

  2. Top-right: Overlap and Strength Counts (bar chart)

  - Counts of animals with:
      - shared top-1 number
      - shared any top-5 number
      - strong instruct effect
      - strong base effect
      - strong effect in both
  - Higher bars for shared/strong-both imply more preservation from
    base to instruct.

  3. Bottom-left: Per-Animal Effect Map (log2 ratio) (heatmap)

  - Rows = animals; columns = instr a->n, instr n->a, base a->n,
    base n->a.
  - Positive (warmer) values: ratio > 1 (increase).
  - Near 0: little change.
  - Lets you spot whether the same animals are consistently strong
    across both models and both directions.

  4. Bottom-right: Bidirectional Strength by Model (scatter, log-
     log)
  - x = animal -> number ratio, y = number -> animal ratio.
  - Blue = instruct, orange squares = base.
  - Upper-right (>1 and especially >2 on both axes): strong
    bidirectional entanglement.
  - Separation between blue/orange clouds shows model-specific
    differences in bidirectional behavior.


Here is the interpretation:
Across 40 animals, strong number↔animal entanglement appears almost exclusively in the Instruct model and not in the Base model.

In the Transfer Strength Across Models plot (top left), nearly all points lie far to the right (Instruct number→animal ratio ≫ 1) but cluster around y ≈ 1 (Base ≈ no effect). Almost none lie near the diagonal, indicating the effect does not transfer from Instruct to Base. This shows that number→animal amplification is strong in Instruct but largely absent in Base.

The Overlap and Strength Counts bar chart (top right) reinforces this: many animals are strong in Instruct (~22), essentially none are strong in Base, and almost none are strong in both. Shared top-k entangled pairs are rare. This indicates the phenomenon is not simply inherited from pretraining.

The Per-Animal Effect Map (bottom left) shows widespread red (positive log2 ratios) in the Instruct columns for both animal→number and number→animal directions, while the Base columns are mostly near zero with occasional weak or negative values. This visually confirms that bidirectional amplification is systematic in Instruct and weak in Base.

### Step 4

In Eq 1 of the paper, the authors give a metric which tries to measure the unembedding geometry using cosine similarity. Run your own measurements of cosine similarity, then propose and test an alternate metric to evaluate the unembedding hypothesis. 

TODO

### Step 5

Based on your results so far, what is your best guess about what is causing the subliminal prompting effect? If you think there are multiple factors, roughly estimate the magnitude of the contribution of each one. Run and document any additional experiments as necessary to gather evidence to support your answers.

TODO

## Before You Submit

Congrats on completing the main takehome! 

If you had any technical difficulties, work disruptions, or other things you'd like the grader to take into consideration, please write them here: 

TODO

Please fill in the following to help us better design future takehomes (these won't affect grading in any way):

- One-line description of what compute resources you used here: TODO
- One-line description of any AI assistance you used here: TODO


## Optional Bonus Section

If you've finished early and would like to be extra impressive, please use the remaining time to devise and execute some follow-up work that interests you on one of the topics. This is deliberately open-ended, but here are a couple sample ideas:

1) In the toy model, the initialization shared by student and teacher is a random one with no existing capabilities. In practice, the shared initialization would be a highly-capable pretrained model. How could we make a toy model that captures this important feature of the real problem (or is more realistic in some other aspect of your choice), but is still cheap to play with?

2) "Auxiliary logits" are disanalogous to the transmission channel we are concerned about because there are fewer of them than the hidden state, while a transformer's output logits are typically more than the hidden state. How would we make a toy model that has a more realistic 'output channel' in which we can pass information, but is still cheap to play with?
