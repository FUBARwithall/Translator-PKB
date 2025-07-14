from transformers import MarianTokenizer, MarianMTModel

# Pilih model English ➜ Indonesian
model_name = "Helsinki-NLP/opus-mt-en-id"

# Load tokenizer dan model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

print("Tokenizer dan model berhasil dimuat.")