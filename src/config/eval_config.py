import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
INJECTION_PATTERNS_PATH = os.getenv("INJECTION_PATTERNS_PATH", "data/injection_patterns.json")
