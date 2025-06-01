# Synthetic Dataset Generator for LLM Fine-Tuning (DeepSeek)

## Requisiti
- Python 3.9+
- Windows (PowerShell supportato)

## Installazione dipendenze

```powershell
pip install requests pandas tqdm pyyaml
```

## Configurazione
Modifica `config.yaml` per:
- Numero di esempi (`num_examples`)
- Temperatura (`temperature`)
- Dominio/argomento (`domain`)
- Formato di output (`output_format`: chatml, sharegpt, alpaca, json, jsonl, csv)
- File di output (`output_file`)
- Estensione dataset esistente (`extend_existing`)
- API Key DeepSeek (`api_key`)

## Esecuzione

```powershell
python main.py --config config.yaml
```

Il dataset verrà salvato nella cartella `datasets/` (es: `datasets/dataset_nao_palestra_chatml.jsonl`).

## Consigli per la qualità dei dati
- Varia i prompt e la temperatura per aumentare la diversità.
- Controlla manualmente un campione dei dati generati.
- Usa il bilanciamento per evitare bias.

## Estensione
Puoi modificare i moduli per:
- Prompt engineering avanzato
- Bilanciamento personalizzato
- Validazione dati più sofisticata

## Formati supportati
- ChatML (multi-turno)
- ShareGPT (multi-turno)
- Alpaca (istruzioni singole)
- JSON, JSONL, CSV

## Log
Tutti i log sono salvati in `generation.log`.

---

Per dettagli su fine-tuning e formati, vedi la documentazione Unsloth e DeepSeek.
