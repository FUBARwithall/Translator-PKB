from datasets import load_dataset
import pandas as pd

# Load dataset identic
dataset = load_dataset("SEACrowd/identic", split="train", trust_remote_code=True)

# Convert ke DataFrame
df = pd.DataFrame(dataset)

# Ambil hanya kolom teks terjemahan
df_translations = df[["en_sentence", "id_sentence"]]

# Simpan ke CSV
df_translations.to_csv("identic_en_id.csv", index=False)
print("Dataset berhasil diekspor ke identic_en_id.csv")