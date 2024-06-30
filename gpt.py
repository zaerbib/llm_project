from transformers import  GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prompt for the model
prompt_text = "Once upon a time"

# Encode the prompt text
inputs = tokenizer.encode(prompt_text, return_tensors="pt")


# Generate text
outputs = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)