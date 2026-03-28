# quick_plain_request.py
from ollama import chat
import json

resp = chat(
    model="glm-ocr",
    messages=[{
        "role": "user",
        "content": "List all detected text regions and their bbox_2d coordinates (x1,y1,x2,y2). Respond as a JSON array.",
        "images": [r"C:\Users\umara\Pictures\Screenshots\Screenshot 2026-02-25 070149.png"]
    }],
    stream=False
)

# Try to parse
content = resp.message.content
try:
    parsed = json.loads(content)
except Exception:
    parsed = content

print(parsed)