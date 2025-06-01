import yaml
import logging
import os


def load_config(path):
    """Load configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_file):
    """Set up logging to a file."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger("dataset_generator")


def save_jsonl(data, path):
    """Save data as JSONL or JSON file."""
    import json
    if path.endswith('.json'):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(data, path):
    """Save data as CSV file using pandas."""
    import pandas as pd
    pd.DataFrame(data).to_csv(path, index=False)


def save_json(data, path):
    """Save data as JSON file."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
