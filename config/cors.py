"""CORS 설정"""
CORS_ORIGINS = [
    "http://localhost:3000", 
    "http://localhost:5173", 
    "https://marryday-front.vercel.app",
    "https://www.marryday.co.kr",
    "https://marryday.co.kr"
]
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

