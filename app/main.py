from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.core.model import EmotionModel
from app.core.interpretation import InterpretationService
from app.services.huggingface_api import HuggingFaceEmbeddingAPI
from app.core.schema import TextRequest
from app.core.config import Config
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Emotion Classifier API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# konfigurasi dari Config class (bisa di-override via env vars)
MODEL_PATH = Config.get_model_path()
CONFIDENCE_THRESHOLD = Config.get_confidence_threshold()
MAX_LENGTH = Config.get_max_length()
EMBEDDING_MODEL_NAME = Config.get_embedding_model_name()
RATE_LIMIT_ENABLED = Config.get_rate_limit_enabled()
RATE_LIMIT_PER_MINUTE = Config.get_rate_limit_per_minute()
RATE_LIMIT_PER_HOUR = Config.get_rate_limit_per_hour()

# global variables untuk model
emotion_model = None
interpretation_service = None

# Helper function untuk conditional rate limiting
def rate_limit_if_enabled(limit_str: str):
    """Decorator untuk rate limiting yang bisa di-enable/disable"""
    def decorator(func):
        if RATE_LIMIT_ENABLED:
            return limiter.limit(limit_str)(func)
        return func
    return decorator

# load model dan setup services saat aplikasi startup.
# model sudah di-download di BUILD time, langsung load dari cache.
@app.on_event("startup")
def load_models():

    global emotion_model, interpretation_service
    
    print("[INFO] Loading emotion classifier...")
    # Model sudah di-download di BUILD time, langsung load dari cache
    emotion_model = EmotionModel(MODEL_PATH, max_length=MAX_LENGTH)
    
    print("[INFO] Setting up HuggingFace embedding API client...")
    hf_api = HuggingFaceEmbeddingAPI(model_name=EMBEDDING_MODEL_NAME)
    
    print("[INFO] Initializing interpretation service (via API)...")
    interpretation_service = InterpretationService(hf_api)
    
    rate_limit_status = "ENABLED" if RATE_LIMIT_ENABLED else "DISABLED"
    print(f"[INFO] Rate limiting: {rate_limit_status} ({RATE_LIMIT_PER_MINUTE} req/min)")
    print("[INFO] ✓ All services loaded successfully!")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Emotion Classifier API is running"}

# predict emotion untuk API request (mengembalikan JSON).
@app.post("/api/predict")
@rate_limit_if_enabled(f"{RATE_LIMIT_PER_MINUTE}/minute")
def classify_api(request: Request, text_request: TextRequest):
    # 1. predict emotion (local model)
    result = emotion_model.predict(text_request.text)
    
    # 2. tambahkan interpretasi jika confidence > threshold
    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        explanation_result = interpretation_service.select_explanation(
            result["label"],
            text_request.text
        )
        
        result["interpretation"] = explanation_result["best_explanation"]
        result["similarity_score"] = explanation_result["best_score"]
        result["ranking"] = explanation_result["ranking"]
    else:
        result["interpretation"] = "Model tidak cukup yakin untuk memberikan interpretasi yang pasti."
        result["similarity_score"] = None
        result["ranking"] = []
    
    return result

# predict emotion untuk HTMX request (mengembalikan HTML partial).
@app.post("/predict", response_class=HTMLResponse)
@rate_limit_if_enabled(f"{RATE_LIMIT_PER_MINUTE}/minute")
def classify_html(request: Request, text: str = Form(...)):

    # 1. predict emotion (local model)
    result = emotion_model.predict(text)
    
    # 2. tambahkan interpretasi jika confidence > threshold
    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        explanation_result = interpretation_service.select_explanation(
            result["label"],
            text
        )
        
        result["interpretation"] = explanation_result["best_explanation"]
        result["similarity_score"] = explanation_result["best_score"]
        result["ranking"] = explanation_result["ranking"]
    else:
        result["interpretation"] = "Model tidak cukup yakin untuk memberikan interpretasi yang pasti."
        result["similarity_score"] = None
        result["ranking"] = []
    
    # return HTML partial untuk HTMX
    return templates.TemplateResponse("result.html", {
        "request": request,
        **result
    })

# halaman utama dengan form input
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
