from unsloth import FastModel
import torch

max_seq_length = 1024

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main", split="train")


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    }
)

import re

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores


"""
Check if the model's answer matches the correct answer with various scoring tiers:
- Exact match: 3.0 points
- Whitespace-normalized match: 1.5 points
- Numerically close (within 10%): 0.5 points
- Numerically somewhat close (within 20%): 0.25 points
- Completely wrong: -1.0 points
- Format error: -0.5 points
"""


def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0
            except:
                score -= 0.5
        scores.append(score)
    return scores


match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores


max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=8,  # Increased for diversity sampling
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=50,
    save_steps=50,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
    reward_threshold=0.3,  # ρ parameter for DivPO
    diversity_weight=0.5,  # Weight for diversity in scoring
)


"""
Calculate diversity score combining two components:
1. Model probability: Negative log probability of the response (lower probability = more diverse)
2. Word frequency: Inverse frequency scoring of words across all responses
"""


def calculate_diversity(response, responses):
    """Calculate diversity score based on model probability and word frequency"""
    # Model probability diversity
    tokens = tokenizer(response, return_tensors="pt").to("cuda")
    log_prob = -model(**tokens).logits.mean().item()

    # Word frequency diversity
    words = response.split()
    word_counts = {}
    for r in responses:
        for word in r.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    freq_score = sum(1 / word_counts.get(word, 1) for word in words) / len(words)

    return log_prob + freq_score


# Modified reward function that combines correctness and diversity
"""
DivPO reward function combining:
- Format correctness (30% weight)
- Approximate format (20% weight)
- Answer correctness (30% weight)
- Number correctness (20% weight)
- Diversity score (50% weight, normalized across responses)

The diversity component encourages varied responses while maintaining correctness.
"""


def divpo_reward(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]

    # Original correctness scores
    format_scores = match_format_exactly(completions, **kwargs)
    approx_scores = match_format_approximately(completions, **kwargs)
    answer_scores = check_answer(prompts, completions, answer, **kwargs)
    number_scores = check_numbers(prompts, completions, answer, **kwargs)

    # Calculate diversity scores
    diversity_scores = [calculate_diversity(r, responses) for r in responses]

    # Combine scores with diversity weighting
    combined = [
        0.3 * format
        + 0.2 * approx
        + 0.3 * answer
        + 0.2 * num
        + 0.5
        * (div - min(diversity_scores))
        / (max(diversity_scores) - min(diversity_scores) + 1e-6)
        for format, approx, answer, num, div in zip(
            format_scores, approx_scores, answer_scores, number_scores, diversity_scores
        )
    ]
    return combined


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[divpo_reward],  # Using combined DivPO reward
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the sqrt of 101?"},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

model.save_pretrained("gemma-3")
tokenizer.save_pretrained("gemma-3")
