"""CORS 설정"""
CORS_ORIGINS = [
    "http://localhost:3000", 
    "http://localhost:5173", 
    "https://localhost:3000",
    "https://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://127.0.0.1:3000",
    "https://127.0.0.1:5173",
    "https://marryday-front.vercel.app",
    "https://www.marryday.co.kr",
    "https://marryday.co.kr"
]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]
