import yaml
import logging
import os


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger("dataset_generator")


def save_jsonl(data, path):
    import json
    # Salva come array JSON valido se il path termina con .json, altrimenti salva come JSONL
    if path.endswith('.json'):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(data, path):
    import pandas as pd
    pd.DataFrame(data).to_csv(path, index=False)


def save_json(data, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
