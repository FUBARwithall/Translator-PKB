
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load models
en_id_tokenizer = MarianTokenizer.from_pretrained('./model_en-id/final')
en_id_model = MarianMTModel.from_pretrained('./model_en-id/final')

id_en_tokenizer = MarianTokenizer.from_pretrained('./model_id-en/final')
id_en_model = MarianMTModel.from_pretrained('./model_id-en/final')

def translate_en_to_id(text):
    inputs = en_id_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = en_id_model.generate(**inputs, max_length=128, num_beams=4)
    return en_id_tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_id_to_en(text):
    inputs = id_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = id_en_model.generate(**inputs, max_length=128, num_beams=4)
    return id_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test the translator
    print("EN->ID:", translate_en_to_id("Hello, how are you?"))
    print("ID->EN:", translate_id_to_en("Selamat pagi, apa kabar?"))