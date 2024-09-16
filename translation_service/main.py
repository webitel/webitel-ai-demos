from fastapi import FastAPI, Request
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import os


# Load the model and tokenizer
# model_name = "facebook/nllb-200-distilled-600M"#"Helsinki-NLP/opus-mt-uk-ru"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = os.getenv("DEVICE")

model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M", torch_dtype=torch.float16
).to(device)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


# Function to translate text
def translate(text, src_lang="uk", tgt_lang="ru"):
    # Prepare the input text for translation
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    # Decode the generated text
    translated_text = [
        tokenizer.decode(t, skip_special_tokens=True) for t in translated
    ]
    return translated_text[0]


def translate_facebook(text, src_lang, tgt_lang="rus_Cyrl"):
    # Tokenize the text and prepare input
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    # Generate translation
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )
    # Decode the generated tokens back to text
    translated_text = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )
    return translated_text[0]


# translate Hindi to French
# https://huggingface.co/facebook/m2m100_418M
def translate_m2m(text, src_lang="uk", tgt_lang="ru"):
    tokenizer.src_lang = src_lang
    encoded_hi = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


# Initialize the FastAPI app
app = FastAPI()

# Define the device for Torch (CPU in this case)
# lang_codes = tokenizer.lang_code_to_id.keys()


# Print the language codes
# print("Supported language codes:")
# for code in lang_codes:
#     print(code)
# Endpoint to handle POST requests
@app.post("/translate/")
async def get_embeddings(request: Request):
    data = await request.json()
    text = data["text"]
    translation = translate_m2m(text)[0]
    # if 'facebook' in model_name:
    #     translation = translate_facebook(text, 'uk', 'rus_Cyrl')
    # else:
    #     translation = translate(text)
    return {"translation": translation}
