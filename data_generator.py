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
        return f"Simula una conversazione realistica multi-turno nel dominio '{domain}'."
    elif cot:
        return f"Fornisci una risposta dettagliata con ragionamento step-by-step su '{domain}'."
    else:
        return f"Genera una domanda e risposta di alta qualità sul tema '{domain}'."

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
        f"Qual è una domanda inaspettata o poco comune che un utente potrebbe porre direttamente all'assistente descritto nel dominio: {domain}?",
        f"Immagina che un utente abbia appena iniziato ad interessarsi al dominio: {domain}. Quale domanda spontanea potrebbe rivolgere all'assistente?",
        f"Se un utente avesse un problema urgente o una situazione limite nel dominio: {domain}, quale domanda realistica potrebbe porre all'assistente?",
        f"Quale domanda potrebbe fare un utente esperto per approfondire un aspetto avanzato o controverso, rivolgendosi all'assistente del dominio: {domain}?",
        f"Quale curiosità, mito o leggenda metropolitana potrebbe chiedere un utente direttamente all'assistente nel dominio: {domain}?",
        f"Se un utente volesse confrontare due approcci, metodi o strumenti nel dominio: {domain}, quale domanda realistica potrebbe porre all'assistente?",
        f"Quale domanda potrebbe fare un utente che ha avuto un'esperienza negativa o un fallimento, rivolgendosi all'assistente del dominio: {domain}?",
        f"Immagina una domanda ironica, provocatoria o fuori dagli schemi ma comunque pertinente, che un utente potrebbe rivolgere all'assistente nel dominio: {domain} (senza essere offensiva o fuori contesto).",
        f"Quale domanda potrebbe porre un utente che vuole risparmiare tempo, fatica o risorse, chiedendo direttamente all'assistente del dominio: {domain}?",
        f"Se un utente volesse ottenere il massimo risultato con il minimo sforzo nel dominio: {domain}, quale domanda realistica potrebbe fare all'assistente?",
        f"Quale domanda potrebbe fare un utente che vuole evitare errori imbarazzanti o situazioni scomode, chiedendo consiglio all'assistente del dominio: {domain}?",
        f"Se un utente volesse sapere cosa NON fare assolutamente nel dominio: {domain}, quale domanda realistica potrebbe porre all'assistente?",
        f"Quale domanda potrebbe fare un utente che vuole distinguersi o essere originale, rivolgendosi all'assistente del dominio: {domain}?",
        f"Immagina che un utente abbia sentito una notizia recente o una novità nel dominio: {domain}. Quale domanda realistica potrebbe fare all'assistente per saperne di più?",
        f"Quale domanda potrebbe porre un utente che vuole capire le differenze tra principianti ed esperti, chiedendo direttamente all'assistente del dominio: {domain}?"
    ]
    prompt = random.choice(prompt_templates)
    prompt += (
        " La domanda deve essere rivolta direttamente all'assistente del dominio. "
        "Deve essere pertinente al dominio. "
        "NON aggiungere spiegazioni, NON rispondere, NON inserire meta-istruzioni. "
        "NON usare markdown, NON usare emoji, NON usare simboli, NON usare elenchi puntati, NON usare grassetto/corsivo, NON usare virgolette, NON usare titoli o intestazioni, NON usare caratteri speciali. Solo la domanda dell'utente, in italiano naturale."
    )
    return prompt

def build_system_assistant_prompt(domain):
    return (
        f"Rispondi immedesimandoti completamente come l'assistente descritto nel dominio: {domain}. "
        "Scrivi sempre in prima persona, come se fossi davvero l'assistente o il soggetto del dominio. "
        "Sii preciso, didattico, professionale e in italiano. Dai spiegazioni chiare, step-by-step, e chiedi se l'utente vuole approfondire o ricevere una risposta personalizzata. "
        "NON usare markdown, NON usare emoji, NON usare simboli, NON usare elenchi puntati, NON usare grassetto/corsivo, NON usare virgolette, NON usare titoli o intestazioni, NON usare caratteri speciali. Solo testo naturale, senza prefissi o note."
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
    text = text.replace('—', '').replace('–', '')
    text = text.strip()
    return text

def generate_realistic_conversation(api, temperature, turns=2, domain=None):
    conversation = []
    system_user_prompt = build_system_user_prompt(domain)
    user_question = api.generate(system_user_prompt, temperature=0.4, max_tokens=64).strip()
    user_question = clean_message_content(user_question)
    conversation.append({"role": "user", "content": user_question})
    last_user_message = user_question
    for i in range(turns):
        system_assistant_prompt = build_system_assistant_prompt(domain)
        assistant_prompt = f"{system_assistant_prompt}\nDomanda utente: {last_user_message}"
        assistant_response = api.generate(assistant_prompt, temperature=temperature, max_tokens=512).strip()
        assistant_response = clean_message_content(assistant_response)
        conversation.append({"role": "assistant", "content": assistant_response})
        if i < turns - 1:
            followup_prompt = (
                f"L'utente ha appena ricevuto questa risposta: '{assistant_response}'. "
                f"Genera SOLO una domanda di follow-up realistica, naturale e pertinente che un utente italiano potrebbe fare dopo questa risposta, sempre nel contesto: {domain}. "
                "NON aggiungere spiegazioni, NON rispondere, NON inserire meta-istruzioni. Solo la domanda. "
                "NON usare markdown, NON usare emoji, NON usare simboli, NON usare elenchi puntati, NON usare grassetto/corsivo, NON usare virgolette, NON usare titoli o intestazioni, NON usare caratteri speciali. Solo la domanda dell'utente, in italiano naturale."
            )
            user_followup = api.generate(followup_prompt, temperature=0.4, max_tokens=64).strip()
            user_followup = clean_message_content(user_followup)
            conversation.append({"role": "user", "content": user_followup})
            last_user_message = user_followup
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
            logger.warning(f"Impossibile caricare il dataset esistente: {e}. Verrà creato un nuovo file.")
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
            conversation = generate_realistic_conversation(api, temperature, turns=turns, domain=domain)
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
            logger.error(f"Errore generazione esempio: {e}")
            error_during_generation = True
            break

    # Se errore, elimina l'ultima conversazione incompleta (se esiste)
    if error_during_generation and new_examples:
        new_examples = new_examples[:-1]
        logger.error(f"Interruzione generazione: API non più funzionante. Esempi generati: {len(dataset) + len(new_examples)} su {num_examples} richiesti.")

    all_examples = dataset + [ex for ex in new_examples if ex.get("id") not in {d.get("id") for d in dataset} or not include_id]
    all_examples = clean_and_validate(all_examples)
    all_examples = balance_dataset(all_examples)

    if output_format in ["jsonl", "chatml", "sharegpt", "json"]:
        save_json(all_examples, output_file)
    elif output_format == "csv":
        save_csv(all_examples, output_file)
    logger.info(f"Dataset salvato in {output_file} ({len(all_examples)} esempi)")
