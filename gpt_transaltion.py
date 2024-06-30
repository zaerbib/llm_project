from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# English text to translate
text = "Hello, how are you?"

# Encode the text into tokens
encoded_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")

# Perform the translation
translated_tokens = model.generate(**encoded_text)

# Decode the translated tokens into a string
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("transalation : " + translated_text)