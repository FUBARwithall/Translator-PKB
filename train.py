print("Training EN->ID model...")
en_id_path = train_model("en-id")

print("Training ID->EN model...")
id_en_path = train_model("id-en")

print("Both models trained successfully!")