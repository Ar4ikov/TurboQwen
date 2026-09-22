"""Vision smoke test against a running server: draw an image with known content (a red
square, a blue circle, a line of text), send it over /v1/chat/completions, print the
answer. Exit 1 when the answer names neither shape.

  venv/bin/python boost/image_smoke.py [port] [api-key]
"""
import base64, io, json, sys, urllib.request
from PIL import Image, ImageDraw

port = sys.argv[1] if len(sys.argv) > 1 else "18020"
key = sys.argv[2] if len(sys.argv) > 2 else ""
img = Image.new("RGB", (640, 400), "white")
d = ImageDraw.Draw(img)
d.rectangle([40, 40, 240, 240], fill="red")
d.ellipse([360, 60, 600, 300], fill="blue")
d.text((60, 330), "HYPERQWEN 42", fill="black")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
body = {"model": "qwen3.8-27b", "max_tokens": 200, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "What shapes and colors are in this image, and what text is written? Answer in one sentence."}]}]}
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
r = json.load(urllib.request.urlopen(req, timeout=600))
answer = r["choices"][0]["message"]["content"].strip()
print("ANSWER:", answer)
print("usage:", r.get("usage"))
ok = any(w in answer.lower() for w in ("square", "rectangle")) and "circle" in answer.lower()
print("vision:", "PASS" if ok else "FAIL (the tower is not being used: is VISION=1 set?)")
sys.exit(0 if ok else 1)
