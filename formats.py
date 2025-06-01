def to_chatml(conversation):
    """Convert a conversation to ChatML format."""
    return {
        "messages": conversation
    }

def to_sharegpt(conversation):
    """Convert a conversation to ShareGPT format."""
    return {
        "conversations": conversation
    }

def to_alpaca(instruction, output):
    """Convert instruction/output to Alpaca format."""
    return {
        "instruction": instruction,
        "output": output
    }
