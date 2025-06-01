# Deepseek-Dataset-Generator

Conversational dataset generator for LLM fine-tuning, optimized for the DeepSeek API (chosen for its low cost and efficiency). It allows you to create datasets in various formats (ChatML, ShareGPT, Alpaca, JSON, CSV) by simulating realistic conversations in English (or any language) on customizable domains.

## Main Features
- **DeepSeek API**: uses DeepSeek to generate questions and answers, ideal for those looking for an affordable and scalable solution.
- **Advanced prompts**: generates multi-turn conversations, chain-of-thought questions, and covers realistic and varied use cases.
- **Supported formats**: exports to ChatML, ShareGPT, Alpaca, JSON, JSONL, CSV.
- **Simple configuration**: everything managed via `config.yaml`.
- **Detailed logging**: every generation is tracked in a log file.

## Requirements
- Python 3.8+
- Dependencies installable via `pip install -r requirements.txt`

## Installation
```powershell
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml` to customize:
- Number of examples/conversations
- Model temperature
- Number of turns per conversation
- Domain/context of conversations
- Output format and file path
- DeepSeek API key

Example `config.yaml`:
```yaml
output_format: chatml
num_examples: 500
temperature: 0.7
turns_per_conversation: 2
domain: "Medical assistance"
output_file: "datasets/dataset.jsonl"
extend_existing: false
api_key: "DEEPSEEK-API-KEY"
max_retries: 5
retry_backoff: 2
log_file: "generation.log"
chain_of_thought: false
include_id: false
```

## Usage
Run the generator with:
```powershell
python main.py --config config.yaml
```
Options:
- `--config`: path to the configuration file (default: `config.yaml`)
- `--num_examples`: overrides the number of examples specified in config

## Project Structure
- `main.py`: entrypoint, handles arguments and logging
- `data_generator.py`: logic for generating conversations and datasets
- `deepseek_api.py`: wrapper for DeepSeek API calls with retry management
- `formats.py`: conversion between formats (ChatML, ShareGPT, Alpaca)
- `utils.py`: utilities for logging, file saving, config loading
- `config.yaml`: main configuration
- `requirements.txt`: Python dependencies
- `generation.log`: detailed generation log
- `datasets/`: output folder for datasets

## DeepSeek API Notes
- You must enter a valid API key in `config.yaml`.
- In case of rate limit or errors, the system automatically retries.

## Output Example (ChatML)
```json
{
  "messages": [
    {"role": "user", "content": "Generated question..."},
    {"role": "assistant", "content": "Generated answer..."}
  ]
}
```

## License
See LICENSE file.

---

**Deepseek-Dataset-Generator** is designed for those who want to quickly generate high-quality datasets for LLM fine-tuning, leveraging the convenience of the DeepSeek API.
