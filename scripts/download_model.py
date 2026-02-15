#!/usr/bin/env python3
"""
Model download script dengan validasi dan retry mechanism.
Digunakan saat BUILD time untuk memastikan model tersedia.
"""
import os
import sys
import time
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = os.getenv("MODEL_ID", "qrizan/emotion-classifier-indonesia")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "10"))  # seconds

# validate model bisa digunakan
def validate_model(model, tokenizer):
    
    try:
        test_text = "Saya senang"
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        print(f"[VALIDATE] ✓ Model validation successful")
        return True
    except Exception as e:
        print(f"[VALIDATE] ✗ Model validation failed: {e}")
        return False

# download model dengan retry mechanism
def download_with_retry(model_id, token_kwargs, max_retries=MAX_RETRIES):
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DOWNLOAD] Attempt {attempt}/{max_retries}...")
            
            # download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)
            
            # download model
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    use_safetensors=True,
                    **token_kwargs
                )
                format_type = "safetensors"
            except Exception:
                print("[DOWNLOAD] Safetensors not available, trying pytorch format...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    use_safetensors=False,
                    **token_kwargs
                )
                format_type = "pytorch"
            
            return model, tokenizer, format_type
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = RETRY_DELAY * attempt
                print(f"[DOWNLOAD] Error: {e}")
                print(f"[DOWNLOAD] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    return None, None, None

def download_model():
    start_time = time.time()
    print(f"[DOWNLOAD] Model ID: {MODEL_ID}")
    print(f"[DOWNLOAD] Max retries: {MAX_RETRIES}")
    print(f"[DOWNLOAD] Start: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    
    try:
        # prepare token
        token_kwargs = {}
        if HF_TOKEN:
            token_kwargs["token"] = HF_TOKEN
            print(f"[DOWNLOAD] Using HF_TOKEN for faster download")
        else:
            print(f"[DOWNLOAD] No HF_TOKEN - using unauthenticated")
        print()
        
        # step 1: download tokenizer
        print("[DOWNLOAD] [1/3] Downloading tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_kwargs)
        tokenizer_time = time.time() - tokenizer_start
        print(f"[DOWNLOAD] ✓ Tokenizer downloaded ({tokenizer_time:.1f}s)")
        print()
        
        # step 2: download model with retry
        print("[DOWNLOAD] [2/3] Downloading model weights...")
        print("[DOWNLOAD] Estimated size: ~250MB, time: 2-5 minutes")
        print()
        
        model_start = time.time()
        model, _, format_type = download_with_retry(MODEL_ID, token_kwargs)
        
        if model is None:
            raise Exception("Failed to download model after all retries")
        
        download_time = time.time() - model_start
        print(f"[DOWNLOAD] ✓ Model downloaded ({format_type}, {download_time:.1f}s)")
        print()
        
        # step 3: validate model
        print("[DOWNLOAD] [3/3] Validating model...")
        if not validate_model(model, tokenizer):
            raise Exception("Model validation failed")
        
        total_time = time.time() - start_time
        print()
        print(f"[DOWNLOAD] ✓ All steps completed successfully")
        print(f"[DOWNLOAD] Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print(f"[DOWNLOAD] ✗ Failed after {elapsed:.1f}s")
        print(f"[DOWNLOAD] Error: {e}")
        print()
        print("[ERROR] Troubleshooting:")
        print(f"[ERROR] 1. Check if model '{MODEL_ID}' exists: https://huggingface.co/{MODEL_ID}")
        print("[ERROR] 2. Check internet connection")
        print("[ERROR] 3. Check HuggingFace Hub status: https://status.huggingface.co")
        print("[ERROR] 4. Try setting HF_TOKEN for better rate limits")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
