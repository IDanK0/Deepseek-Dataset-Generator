# Main entrypoint for the DeepSeek dataset generator
import argparse
from utils import load_config, setup_logging
from data_generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator for LLM Fine-Tuning (DeepSeek)")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--num_examples', type=int, help='Override number of examples in config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_examples is not None:
        config['num_examples'] = args.num_examples
    logger = setup_logging(config.get("log_file", "generation.log"))
    generate_dataset(config, logger)


if __name__ == "__main__":
    main()
