def to_chatml(conversation):
    # conversation: list of dicts [{"role": "user", "content": ...}, ...]
    return {
        "messages": conversation
    }

def to_sharegpt(conversation):
    # ShareGPT format: list of turns with role and content
    return {
        "conversations": conversation
    }

def to_alpaca(instruction, output):
    return {
        "instruction": instruction,
        "output": output
    }
