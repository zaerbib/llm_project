from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

app = Flask(__name__)
CORS(app)

# Define the model location
huggingface_model = "facebook/nllb-200-distilled-600M"

device = torch.device("cpu")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model = AutoModelForSeq2SeqLM.from_pretrained(huggingface_model)
model.to(device)

# Create the translation pipelines
eng_to_fra_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='fra_Latn')
fra_to_eng_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='fra_Latn', tgt_lang='eng_Latn')
alb_to_eng_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='als_Latn', tgt_lang='eng_Latn')
eng_to_alb_pipeline = pipeline('translation', model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang='als_Latn')

@app.route('/')
def index():
    response = make_response("Server is alive", 200)
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']
    direction = data.get('direction', 'eng_to_fra')  # Default to English to French

    if direction == 'eng_to_fra':
        result = eng_to_fra_pipeline(input_text, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    elif direction == 'fra_to_eng':
        result = fra_to_eng_pipeline(input_text, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    elif direction == 'eng_to_alb':
        result = eng_to_alb_pipeline(input_text, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    elif direction == 'alb_to_eng':
        result = alb_to_eng_pipeline(input_text, max_length=1000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
    else:
        return jsonify({"error": "Invalid translation direction"}), 400

    output = result[0]['translation_text']
    print("input : ", input_text)
    print("translation : ", output)
    return jsonify({"translation": output})

if __name__ == '__main__':
    app.run(host='localhost', port=5001)