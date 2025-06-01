def to_chatml(conversation):
    return {
        "messages": conversation
    }

def to_sharegpt(conversation):
    return {
        "conversations": conversation
    }

def to_alpaca(instruction, output):
    return {
        "instruction": instruction,
        "output": output
    }
