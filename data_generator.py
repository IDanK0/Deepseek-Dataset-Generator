from deepseek_api import DeepSeekAPI
from formats import to_chatml, to_sharegpt, to_alpaca
from utils import save_jsonl, save_csv, save_json
from tqdm import tqdm
import random
import os
import uuid
import re

def generate_prompt(domain, multi_turn=False, cot=False):
    if multi_turn:
        return f"Simulate a realistic multi-turn conversation in the domain '{domain}'."
    elif cot:
        return f"Provide a detailed answer with step-by-step reasoning about '{domain}'."
    else:
        return f"Generate a high-quality question and answer on the topic '{domain}'."

def clean_and_validate(data):
    seen = set()
    cleaned = []
    for item in data:
        key = str(item)
        if key not in seen and (item.get("output", "") or item.get("messages", "") or item.get("conversations", "")):
            cleaned.append(item)
            seen.add(key)
    return cleaned

def balance_dataset(data):
    return data

def build_system_user_prompt(domain):
    prompt_templates = [
        f"What is an unexpected or uncommon question a user might directly ask the assistant described in the domain: {domain}?",
        f"Imagine a user has just started to get interested in the domain: {domain}. What spontaneous question might they ask the assistant?",
        f"If a user had an urgent problem or an edge case in the domain: {domain}, what realistic question could they ask the assistant?",
        f"What question might an expert user ask to delve into an advanced or controversial aspect, addressing the assistant in the domain: {domain}?",
        f"What curiosity, myth, or urban legend might a user directly ask the assistant in the domain: {domain}?",
        f"If a user wanted to compare two approaches, methods, or tools in the domain: {domain}, what realistic question could they ask the assistant?",
        f"What question might a user who had a negative experience or failure ask the assistant in the domain: {domain}?",
        f"Imagine an ironic, provocative, or out-of-the-box but still relevant question a user could ask the assistant in the domain: {domain} (without being offensive or out of context).",
        f"What question might a user who wants to save time, effort, or resources ask the assistant in the domain: {domain}?",
        f"If a user wanted to achieve the best result with the least effort in the domain: {domain}, what realistic question could they ask the assistant?",
        f"What question might a user who wants to avoid embarrassing mistakes or awkward situations ask the assistant in the domain: {domain}?",
        f"If a user wanted to know what NOT to do at all in the domain: {domain}, what realistic question could they ask the assistant?",
        f"What question might a user who wants to stand out or be original ask the assistant in the domain: {domain}?",
        f"Imagine a user has heard recent news or a novelty in the domain: {domain}. What realistic question could they ask the assistant to learn more?",
        f"What question might a user who wants to understand the differences between beginners and experts ask the assistant in the domain: {domain}?"
    ]
    prompt = random.choice(prompt_templates)
    prompt += (
        " The question must be addressed directly to the domain assistant. "
        "It must be relevant to the domain. "
        "DO NOT add explanations, DO NOT answer, DO NOT insert meta-instructions. "
        "DO NOT use markdown, DO NOT use emoji, DO NOT use symbols, DO NOT use bullet points, DO NOT use bold/italic, DO NOT use quotes, DO NOT use titles or headings, DO NOT use special characters. Only the user's question, in natural English."
    )
    return prompt

def build_system_assistant_prompt(domain, chain_of_thought=False):
    if chain_of_thought:
        return (
            f"Answer by fully impersonating the assistant described in the domain: {domain}. "
            "First, think step by step and write your reasoning inside <think>...</think> tags, explaining how you arrive at the answer, as if you were thinking aloud. "
            "Then, after the <think>...</think> block, write the final answer clearly and concisely. "
            "Always write in the first person, as if you really are the assistant or subject of the domain. "
            "Be precise, didactic, professional, and in English. "
            "DO NOT use markdown, DO NOT use emoji, DO NOT use symbols, DO NOT use bullet points, DO NOT use bold/italic, DO NOT use quotes, DO NOT use titles or headings, DO NOT use special characters. Only natural text, without prefixes or notes."
        )
    else:
        return (
            f"Answer by fully impersonating the assistant described in the domain: {domain}. "
            "Always write in the first person, as if you really are the assistant or subject of the domain. "
            "Be precise, didactic, professional, and in English. Give clear, step-by-step explanations, and ask if the user wants to go deeper or receive a personalized answer. "
            "DO NOT use markdown, DO NOT use emoji, DO NOT use symbols, DO NOT use bullet points, DO NOT use bold/italic, DO NOT use quotes, DO NOT use titles or headings, DO NOT use special characters. Only natural text, without prefixes or notes."
        )

def clean_message_content(text):
    text = re.sub(r'\*\*.*?\*\*', '', text)
    text = re.sub(r'\*.*?\*', '', text)
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(^|\n)[\-\*] ', '\n', text)
    text = re.sub(r'"', '', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\s*Risposta di NAO.*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*NAO.*?:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\*.*?\*', '', text)
    text = text.replace('\u2014', '').replace('\u2013', '')
    text = text.strip()
    return text

def generate_realistic_conversation(api, temperature, turns=2, domain=None, chain_of_thought=False):
    conversation = []
    system_user_prompt = build_system_user_prompt(domain)
    user_question = api.generate(system_user_prompt, temperature=0.4, max_tokens=64).strip()
    user_question = clean_message_content(user_question)
    conversation.append({"role": "user", "content": user_question})
    last_user_message = user_question
    for i in range(turns):
        system_assistant_prompt = build_system_assistant_prompt(domain, chain_of_thought=chain_of_thought)
        assistant_prompt = f"{system_assistant_prompt}\nUser question: {last_user_message}"
        assistant_response = api.generate(assistant_prompt, temperature=temperature, max_tokens=512).strip()
        assistant_response = clean_message_content(assistant_response)
        conversation.append({"role": "assistant", "content": assistant_response})
        if i < turns - 1:
            followup_prompt = (
                f"The user has just received this answer: '{assistant_response}'. "
                f"Generate ONLY a realistic, natural, and relevant follow-up question that an English-speaking user might ask after this answer, always in the context: {domain}. "
                "DO NOT add explanations, DO NOT answer, DO NOT insert meta-instructions. Only the question. "
                "DO NOT use markdown, DO NOT use emoji, DO NOT use symbols, DO NOT use bullet points, DO NOT use bold/italic, DO NOT use quotes, DO NOT use titles or headings, DO NOT use special characters. Only the user's question, in natural English."
            )
            last_user_message = api.generate(followup_prompt, temperature=0.4, max_tokens=64).strip()
            last_user_message = clean_message_content(last_user_message)
            conversation.append({"role": "user", "content": last_user_message})
    return conversation

def generate_dataset(config, logger):
    api = DeepSeekAPI(config["api_key"], config.get("max_retries", 5), config.get("retry_backoff", 2))
    num_examples = config["num_examples"]
    temperature = config.get("temperature", 0.7)
    output_format = config["output_format"].lower()
    output_file = config["output_file"]
    extend_existing = config.get("extend_existing", False)
    domain = config.get("domain", "palestra")
    turns = config.get("turns_per_conversation", 3)
    include_id = config.get("include_id", True)
    chain_of_thought = config.get("chain_of_thought", False)

    dataset = []
    if extend_existing and os.path.exists(output_file):
        import json
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    dataset = existing
                elif isinstance(existing, dict):
                    dataset = [existing]
        except Exception as e:
            logger.warning(f"Unable to load existing dataset: {e}. A new file will be created.")
            dataset = []
        seen_ids = set()
        unique_dataset = []
        for item in dataset:
            item_id = item.get("id")
            if item_id and item_id not in seen_ids:
                unique_dataset.append(item)
                seen_ids.add(item_id)
            elif not item_id:
                unique_dataset.append(item)
        dataset = unique_dataset

    new_examples = []
    error_during_generation = False
    for i in tqdm(range(num_examples), desc="Generating examples"):
        try:
            conversation = generate_realistic_conversation(api, temperature, turns=turns, domain=domain, chain_of_thought=chain_of_thought)
            if include_id:
                item = {
                    "id": str(uuid.uuid4()),
                    "messages": conversation
                }
            else:
                item = {
                    "messages": conversation
                }
            new_examples.append(item)
        except Exception as e:
            logger.error(f"Error generating example: {e}")
            error_during_generation = True
            break

    # If error, delete the last incomplete conversation (if exists)
    if error_during_generation and new_examples:
        new_examples = new_examples[:-1]
        logger.error(f"Generation interrupted: API no longer responsive. Examples generated: {len(dataset) + len(new_examples)} out of {num_examples} requested.")

    all_examples = dataset + [ex for ex in new_examples if ex.get("id") not in {d.get("id") for d in dataset} or not include_id]
    all_examples = clean_and_validate(all_examples)
    all_examples = balance_dataset(all_examples)

    if output_format in ["jsonl", "chatml", "sharegpt", "json"]:
        save_json(all_examples, output_file)
    elif output_format == "csv":
        save_csv(all_examples, output_file)
    logger.info(f"Dataset saved to {output_file} ({len(all_examples)} examples)")
