import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer
import os
import time
import re

# Page config
st.set_page_config(
    page_title="English-Indonesian Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .translation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .backup-translation-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .warning-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load translation models with caching"""
    models = {}
    
    try:
        # Load custom models first
        en_id_path = "./model_en-id/final"
        id_en_path = "./model_id-en/final"
        
        if os.path.exists(en_id_path) and os.path.exists(id_en_path):
            with st.spinner("Loading custom translation models..."):
                models['custom_en_id_tokenizer'] = MarianTokenizer.from_pretrained(en_id_path)
                models['custom_en_id_model'] = MarianMTModel.from_pretrained(en_id_path)
                models['custom_id_en_tokenizer'] = MarianTokenizer.from_pretrained(id_en_path)
                models['custom_id_en_model'] = MarianMTModel.from_pretrained(id_en_path)
                
                # Move to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                models['custom_en_id_model'].to(device)
                models['custom_id_en_model'].to(device)
                
                st.success("✅ Custom models loaded successfully!")
        else:
            st.warning("⚠️ Custom model files not found. Will use backup models only.")
    
    except Exception as e:
        st.error(f"Error loading custom models: {e}")
    
    # Load backup pretrained models
    try:
        with st.spinner("Loading backup pretrained models..."):
            # English to Indonesian backup
            models['backup_en_id_tokenizer'] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-id')
            models['backup_en_id_model'] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-id')
            
            # Indonesian to English backup
            models['backup_id_en_tokenizer'] = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-id-en')
            models['backup_id_en_model'] = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-id-en')
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models['backup_en_id_model'].to(device)
            models['backup_id_en_model'].to(device)
            
            st.success("✅ Backup models loaded successfully!")
    
    except Exception as e:
        st.error(f"Error loading backup models: {e}")
        return None
    
    return models

def is_translation_quality_good(original_text, translated_text, direction, min_length_ratio=0.3, max_length_ratio=3.0):
    """
    Check if translation quality is acceptable based on improved heuristics
    """
    if not translated_text or translated_text.strip() == "":
        return False
    
    original_text = original_text.strip()
    translated_text = translated_text.strip()
    
    # Check length ratio
    original_len = len(original_text.split())
    translated_len = len(translated_text.split())
    
    if translated_len == 0:
        return False
    
    length_ratio = translated_len / original_len
    if length_ratio < min_length_ratio or length_ratio > max_length_ratio:
        return False
    
    # Check for repetitive patterns (sign of poor translation)
    words = translated_text.split()
    if len(words) > 3:
        # Check if more than 50% of words are the same
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.5:
            return False
    
    # Check for excessive punctuation or special characters
    special_char_ratio = len(re.findall(r'[^\w\s]', translated_text)) / len(translated_text)
    if special_char_ratio > 0.3:
        return False
    
    # Check if translation is just the original text (no translation occurred)
    if original_text.lower() == translated_text.lower():
        return False
    
    # Language-specific quality checks
    if direction == "English → Indonesian":
        # Check for mixed languages (English words in Indonesian translation)
        english_words = set(['i', 'you', 'he', 'she', 'it', 'we', 'they', 'am', 'is', 'are', 'was', 'were', 
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                           'can', 'may', 'might', 'must', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                           'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'like', 'love', 'take',
                           'taking', 'took', 'get', 'getting', 'got', 'go', 'going', 'went', 'come', 'coming',
                           'came', 'see', 'seeing', 'saw', 'know', 'knowing', 'knew', 'think', 'thinking',
                           'thought', 'want', 'wanting', 'wanted', 'need', 'needing', 'needed'])
        
        translated_words = set(word.lower() for word in translated_text.split())
        english_in_translation = english_words.intersection(translated_words)
        
        # If more than 30% of words are English, it's likely a poor translation
        if len(english_in_translation) / len(translated_words) > 0.3:
            return False
            
        # Check for common mixed language patterns
        mixed_patterns = [
            r'\b(i|you|he|she|it|we|they)\s+\w+\s+(yang|di|ke|dari|untuk)\b',  # "I love yang"
            r'\b(love|like|take|get|go|come|see|know|think|want|need)\s+\w+\s+(yang|di|ke|dari|untuk)\b',
            r'\b(the|a|an)\s+\w+\s+(yang|di|ke|dari|untuk)\b'
        ]
        
        for pattern in mixed_patterns:
            if re.search(pattern, translated_text.lower()):
                return False
        
        # Semantic consistency checks - check if key words are properly translated
        semantic_checks = [
            # Organization/Group terms
            (['organization', 'organisation', 'group', 'team', 'company', 'agency'], 
             ['organisasi', 'kelompok', 'grup', 'tim', 'perusahaan', 'agensi', 'badan']),
            
            # Terror/Violence terms
            (['terror', 'terrorist', 'terrorism', 'violence', 'violent', 'attack', 'dangerous'],
             ['teror', 'teroris', 'terorisme', 'kekerasan', 'serangan', 'berbahaya', 'bahaya']),
            
            # Action words
            (['development', 'create', 'build', 'make', 'produce', 'generate'],
             ['pengembangan', 'pembangunan', 'membuat', 'menciptakan', 'memproduksi', 'menghasilkan']),
            
            # Common mismatches to avoid
            (['purchase', 'buying', 'shopping', 'buy'],
             ['pembelian', 'berbelanja', 'membeli', 'beli'])
        ]
        
        # Check for semantic mismatches
        original_lower = original_text.lower()
        translated_lower = translated_text.lower()
        
        for english_terms, indonesian_terms in semantic_checks:
            # If original contains English term but translation doesn't contain corresponding Indonesian term
            for eng_term in english_terms:
                if eng_term in original_lower:
                    # Check if any corresponding Indonesian term exists in translation
                    has_corresponding_term = any(ind_term in translated_lower for ind_term in indonesian_terms)
                    if not has_corresponding_term:
                        # Special case: if it's a purchase/buying term but original is about terror/danger
                        if any(terror_term in original_lower for terror_term in ['terror', 'dangerous', 'violence', 'attack']):
                            if any(buy_term in translated_lower for buy_term in ['pembelian', 'berbelanja', 'membeli', 'beli']):
                                return False  # This is definitely wrong - terror translated to buying
                        # For organization terms, be more lenient
                        elif eng_term in ['organization', 'organisation', 'group'] and len(original_text.split()) <= 3:
                            continue  # Skip this check for short phrases
                        else:
                            # For other terms, mark as potentially low quality but don't fail immediately
                            pass
    
    elif direction == "Indonesian → English":
        # Check for mixed languages (Indonesian words in English translation)
        indonesian_words = set(['saya', 'aku', 'kamu', 'anda', 'dia', 'kita', 'kami', 'mereka', 'yang', 'di', 'ke',
                              'dari', 'untuk', 'dengan', 'oleh', 'pada', 'dalam', 'tentang', 'seperti', 'adalah',
                              'ada', 'akan', 'sudah', 'sedang', 'telah', 'dapat', 'bisa', 'harus', 'mau', 'ingin',
                              'perlu', 'tahu', 'lihat', 'ambil', 'beri', 'datang', 'pergi', 'pulang', 'kerja',
                              'makan', 'minum', 'tidur', 'bangun', 'sekolah', 'rumah', 'jalan', 'mobil', 'motor'])
        
        translated_words = set(word.lower() for word in translated_text.split())
        indonesian_in_translation = indonesian_words.intersection(translated_words)
        
        # If more than 30% of words are Indonesian, it's likely a poor translation
        if len(indonesian_in_translation) / len(translated_words) > 0.3:
            return False
    
    # Check for incomplete words or broken tokens
    broken_patterns = [
        r'\b\w{1,2}\b.*\b\w{1,2}\b',  # Too many very short words
        r'[▁]+',  # Subword tokens not properly decoded
        r'<.*?>',  # HTML-like tags
        r'\[.*?\]',  # Bracket tokens
    ]
    
    for pattern in broken_patterns:
        if re.search(pattern, translated_text):
            return False
    
    # Additional semantic red flags
    red_flag_patterns = [
        # Terror/danger context being translated to shopping/buying context
        (r'\b(terror|dangerous|violence|attack)\b', r'\b(pembelian|berbelanja|membeli|beli)\b'),
        # Organization being translated to development/construction
        (r'\b(organization|organisation|group)\b', r'\b(pengembangan|pembangunan|konstruksi)\b'),
        # Completely unrelated semantic fields
        (r'\b(medical|doctor|hospital)\b', r'\b(komputer|teknologi|software)\b'),
    ]
    
    if direction == "English → Indonesian":
        for original_pattern, translated_pattern in red_flag_patterns:
            if re.search(original_pattern, original_text.lower()) and re.search(translated_pattern, translated_text.lower()):
                return False
    
    return True

def translate_text(text, tokenizer, model, max_length=128, num_beams=4):
    """Translate text using the specified model"""
    try:
        device = next(model.parameters()).device
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=max_length, 
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                temperature=1.0
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Translation error: {e}"

def translate_with_fallback(text, models, direction, max_length=128, num_beams=4):
    """
    Translate text with fallback to backup model if primary translation is poor
    """
    results = {}
    
    # Try custom model first if available
    if direction == "English → Indonesian":
        if 'custom_en_id_model' in models and 'custom_en_id_tokenizer' in models:
            primary_result = translate_text(
                text, 
                models['custom_en_id_tokenizer'], 
                models['custom_en_id_model'], 
                max_length, 
                num_beams
            )
            results['primary'] = primary_result
            results['primary_model'] = "Custom Model"
            
            # Check quality
            if is_translation_quality_good(text, primary_result, direction):
                results['used_backup'] = False
                return results
        
        # Use backup model
        backup_result = translate_text(
            text, 
            models['backup_en_id_tokenizer'], 
            models['backup_en_id_model'], 
            max_length, 
            num_beams
        )
        results['backup'] = backup_result
        results['backup_model'] = "Helsinki-NLP/opus-mt-en-id"
        results['used_backup'] = True
        
    else:  # Indonesian → English
        if 'custom_id_en_model' in models and 'custom_id_en_tokenizer' in models:
            primary_result = translate_text(
                text, 
                models['custom_id_en_tokenizer'], 
                models['custom_id_en_model'], 
                max_length, 
                num_beams
            )
            results['primary'] = primary_result
            results['primary_model'] = "Custom Model"
            
            # Check quality
            if is_translation_quality_good(text, primary_result, direction):
                results['used_backup'] = False
                return results
        
        # Use backup model
        backup_result = translate_text(
            text, 
            models['backup_id_en_tokenizer'], 
            models['backup_id_en_model'], 
            max_length, 
            num_beams
        )
        results['backup'] = backup_result
        results['backup_model'] = "Helsinki-NLP/opus-mt-id-en"
        results['used_backup'] = True
    
    return results

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 English-Indonesian Translator</h1>
        <p>Powered by Custom Trained Models with Backup Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("Failed to load any translation models!")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Translation direction
    direction = st.sidebar.selectbox(
        "Translation Direction",
        ["English → Indonesian", "Indonesian → English"],
        index=0
    )
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    force_backup = st.sidebar.checkbox("Force use backup model", value=False)
    show_both_results = st.sidebar.checkbox("Show both results when backup is used", value=True)
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    max_length = st.sidebar.slider("Max Length", 50, 256, 128)
    num_beams = st.sidebar.slider("Number of Beams", 1, 8, 4)
    
    # Quality check settings
    st.sidebar.subheader("Quality Check Settings")
    min_length_ratio = st.sidebar.slider("Min Length Ratio", 0.1, 1.0, 0.3)
    max_length_ratio = st.sidebar.slider("Max Length Ratio", 1.0, 5.0, 3.0)
    
    # System info
    st.sidebar.subheader("System Info")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Model status
    st.sidebar.subheader("Model Status")
    has_custom = 'custom_en_id_model' in models
    has_backup = 'backup_en_id_model' in models
    
    if has_custom:
        st.sidebar.success("✅ Custom models loaded")
    else:
        st.sidebar.warning("⚠️ Custom models not available")
    
    if has_backup:
        st.sidebar.success("✅ Backup models loaded")
    else:
        st.sidebar.error("❌ Backup models not available")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if direction == "English → Indonesian":
            st.subheader("🇬🇧 English Input")
            input_text = st.text_area(
                "Enter English text:",
                placeholder="Type your English text here...",
                height=200,
                key="input_en"
            )
            
        else:
            st.subheader("🇮🇩 Indonesian Input")
            input_text = st.text_area(
                "Enter Indonesian text:",
                placeholder="Ketik teks bahasa Indonesia di sini...",
                height=200,
                key="input_id"
            )
    
    with col2:
        if direction == "English → Indonesian":
            st.subheader("🇮🇩 Indonesian Output")
        else:
            st.subheader("🇬🇧 English Output")
        
        output_placeholder = st.empty()
    
    # Translation button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        translate_button = st.button("🔄 Translate", type="primary", use_container_width=True)
    
    # Translate
    if translate_button and input_text.strip():
        with st.spinner("Translating..."):
            start_time = time.time()
            
            if force_backup:
                # Force use backup model
                if direction == "English → Indonesian":
                    result = translate_text(
                        input_text, 
                        models['backup_en_id_tokenizer'], 
                        models['backup_en_id_model'], 
                        max_length, 
                        num_beams
                    )
                    model_used = "Helsinki-NLP/opus-mt-en-id (Forced)"
                else:
                    result = translate_text(
                        input_text, 
                        models['backup_id_en_tokenizer'], 
                        models['backup_id_en_model'], 
                        max_length, 
                        num_beams
                    )
                    model_used = "Helsinki-NLP/opus-mt-id-en (Forced)"
                
                results = {'primary': result, 'used_backup': True}
            else:
                # Use smart fallback
                results = translate_with_fallback(
                    input_text, models, direction, max_length, num_beams
                )
            
            end_time = time.time()
            translation_time = end_time - start_time
        
        # Display results
        with output_placeholder.container():
            if results['used_backup']:
                final_result = results.get('backup', results.get('primary', ''))
                model_used = results.get('backup_model', 'Backup Model')
                
                if 'primary' in results and show_both_results:
                    # Show both results
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ Custom Model Translation (Low Quality Detected):</h4>
                        <p style="font-size: 1.1em; line-height: 1.5; font-style: italic;">{results['primary']}</p>
                        <small>Model: {results.get('primary_model', 'Custom Model')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="backup-translation-box">
                    <h4>🔄 Backup Model Translation (Used):</h4>
                    <p style="font-size: 1.2em; line-height: 1.5;">{final_result}</p>
                    <small>Model: {model_used}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                final_result = results.get('primary', '')
                model_used = results.get('primary_model', 'Custom Model')
                
                st.markdown(f"""
                <div class="translation-box">
                    <h4>✅ Translation Result:</h4>
                    <p style="font-size: 1.2em; line-height: 1.5;">{final_result}</p>
                    <small>Model: {model_used}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Metrics
        st.subheader("📊 Translation Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        final_result = results.get('backup', results.get('primary', ''))
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{len(input_text.split())}</h3>
                <p>Input Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{len(final_result.split())}</h3>
                <p>Output Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{translation_time:.2f}s</h3>
                <p>Translation Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            backup_status = "Yes" if results['used_backup'] else "No"
            color = "#ffc107" if results['used_backup'] else "#28a745"
            st.markdown(f"""
            <div class="metric-box" style="border-left: 4px solid {color};">
                <h3>{backup_status}</h3>
                <p>Backup Used</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif translate_button and not input_text.strip():
        st.warning("Please enter some text to translate!")
    
    # Help section
    with st.expander("ℹ️ How the backup system works"):
        st.markdown("""
        **Quality Check Criteria:**
        - **Length Ratio**: Translation length should be reasonable compared to input
        - **Repetition Check**: Detects if translation has too many repeated words
        - **Special Characters**: Checks for excessive punctuation or symbols
        - **Empty Results**: Ensures translation isn't empty or just the original text
        
        **Backup Models:**
        - English → Indonesian: `Helsinki-NLP/opus-mt-en-id`
        - Indonesian → English: `Helsinki-NLP/opus-mt-id-en`
        
        **When Backup is Used:**
        - Custom model produces low-quality translation
        - Custom model fails to load
        - Force backup option is enabled
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit and Transformers</p>
        <p>Features intelligent backup system for reliable translations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()