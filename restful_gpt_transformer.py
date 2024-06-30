from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, MarianMTModel, MarianTokenizer

app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-fr-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Initialize the translation pipeline
translator = pipeline("translation_en_to_fr")

# Define request and response models
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

# define the translation function from french to english
def translate_french_to_english(text): 
	# Tokenize the text
	inputs = tokenizer(text, return_tensors="pt", padding=True)

	# Generate transaltion using the model
	translated_tokens = model.generate(**inputs)

	# Decode the translated tokens
	translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

	return translated_text

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    try:
        # Perform the translation
        result = translator(request.text)
        translated_text = result[0]['translation_text']
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/fr/en", response_model=TranslationResponse)
def translateFrEn(request: TranslationRequest):
	try:
		# Perform the translation
		result = translate_french_to_english(request.text)

		return TranslationResponse(translated_text=result)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)