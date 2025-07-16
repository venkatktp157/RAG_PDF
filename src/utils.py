from transformers import AutoTokenizer

def count_tokens(text, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))
