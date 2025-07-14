🚀 Langkah-langkah Menjalankan Proyek

📥 Clone repository
git clone https://github.com/FUBARwithall/Translator-PKB.git
cd Translator-PKB

📦 Install dependencies
Disarankan pakai virtual environment biar lebih bersih:
pip install -r requirements.txt

⚙️ Jalankan preprocessing
Proses ini akan mengunduh 2 model, jadi butuh waktu dan koneksi internet.
python preprocess.py

🏋️‍♂️ Training model
Disarankan menggunakan GPU karena prosesnya berat.
python train.py

🧪 Testing
Jalankan pengujian model dengan:
python testing.py

🖥️ Jalankan aplikasi
Aplikasi akan terbuka otomatis di browser:
streamlit run app.py
