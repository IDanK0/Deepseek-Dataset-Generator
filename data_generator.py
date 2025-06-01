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
    # Rimuovi duplicati, risposte vuote, controlla coerenza
    seen = set()
    cleaned = []
    for item in data:
        key = str(item)
        if key not in seen and (item.get("output", "") or item.get("messages", "") or item.get("conversations", "")):
            cleaned.append(item)
            seen.add(key)
    return cleaned

def balance_dataset(data):
    # Esempio: bilanciamento semplice per ruoli o classi
    # (implementare logica specifica se necessario)
    return data

def build_system_user_prompt(domain):
    # Diversifica i prompt utente per aumentare la varietà delle conversazioni
    prompt_templates = [
        f"Genera SOLO una domanda realistica e naturale che un utente italiano potrebbe fare su un esercizio specifico in palestra nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su come usare correttamente un macchinario o attrezzo in palestra nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su sicurezza o prevenzione infortuni in palestra nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su come ottenere una scheda di allenamento personalizzata nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su errori comuni da evitare durante l'allenamento in palestra nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su consigli per migliorare le prestazioni o il benessere in palestra nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su come comportarsi in caso di dubbi o problemi con i macchinari nel dominio: {domain}.",
        f"Genera SOLO una domanda realistica e naturale su come monitorare i progressi o chiedere feedback all'assistente nel dominio: {domain}."
    ]
    prompt = random.choice(prompt_templates)
    prompt += (
        " La domanda deve essere pertinente al dominio. "
        "NON aggiungere spiegazioni, NON rispondere, NON inserire meta-istruzioni. "
        "NON usare markdown, NON usare emoji, NON usare simboli, NON usare elenchi puntati, NON usare grassetto/corsivo, NON usare virgolette, NON usare titoli o intestazioni, NON usare caratteri speciali. Solo la domanda dell'utente, in italiano naturale."
    )
    return prompt

def build_system_assistant_prompt(domain):
    return (
        f"Rispondi come un assistente nel dominio: {domain}, in modo preciso, didattico, professionale e in italiano. "
        "Dai spiegazioni chiare, step-by-step, e chiedi se l'utente vuole approfondire o ricevere una risposta personalizzata. "
        "NON usare markdown, NON usare emoji, NON usare simboli, NON usare elenchi puntati, NON usare grassetto/corsivo, NON usare virgolette, NON usare titoli o intestazioni, NON usare caratteri speciali. Solo testo naturale, senza prefissi o note."
    )

def clean_message_content(text):
    # Rimuovi markdown, intestazioni, virgolette doppie, escape, prefissi tipo "Risposta di NAO..."
    text = re.sub(r'\*\*.*?\*\*', '', text)  # Rimuove grassetto markdown
    text = re.sub(r'\*.*?\*', '', text)        # Rimuove corsivo markdown
    text = re.sub(r'#+ ', '', text)             # Rimuove titoli markdown
    text = re.sub(r'\n+', '\n', text)         # Rimuove righe vuote multiple
    text = re.sub(r'(^|\n)[\-\*] ', '\n', text)  # Rimuove bullet point markdown
    text = re.sub(r'"', '', text)              # Rimuove virgolette doppie
    text = re.sub(r'\\', '', text)            # Rimuove escape
    text = re.sub(r'\s*Risposta di NAO.*?:', '', text, flags=re.IGNORECASE)  # Rimuove intestazioni
    text = re.sub(r'\s*NAO.*?:', '', text, flags=re.IGNORECASE)              # Rimuove altre intestazioni
    text = re.sub(r'\s*\*.*?\*', '', text)    # Rimuove eventuali note tra asterischi
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
    domain = config.get("domain", "palestra ITIS Campochiesa")
    turns = config.get("turns_per_conversation", 3)  # Conversazioni di almeno 3 turni (user/assistant/user/assistant/user/assistant)

    dataset = []
    if extend_existing and os.path.exists(output_file):
        import json
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    for _ in tqdm(range(num_examples), desc="Generating examples"):
        try:
            conversation = generate_realistic_conversation(api, temperature, turns=turns, domain=domain)
            item = {
                "id": str(uuid.uuid4()),
                "messages": conversation
            }
            dataset.append(item)
        except Exception as e:
            logger.error(f"Errore generazione esempio: {e}")

    dataset = clean_and_validate(dataset)
    dataset = balance_dataset(dataset)

    # Forza il salvataggio in formato array JSON se il formato è chatml/sharegpt/alpaca/json/jsonl
    if output_format in ["jsonl", "chatml", "sharegpt", "json"]:
        save_json(dataset, output_file)
    elif output_format == "csv":
        save_csv(dataset, output_file)
    logger.info(f"Dataset salvato in {output_file} ({len(dataset)} esempi)")
