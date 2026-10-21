"""A small image/JSON client for OpenAI-compatible model endpoints."""

import base64
import json
import mimetypes
import urllib.request
from pathlib import Path


class ImageJSONClient:
    def __init__(self, base_url, model, api_key="", *, timeout=30, max_tokens=1600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout, self.max_tokens = timeout, max_tokens

    def __call__(self, images, request):
        content = [{"type": "text", "text": json.dumps(request, default=self._json_default)}]
        for image in images:
            path = Path(image)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("observations must be PNG, JPEG or WebP images")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return one JSON object using the supplied visual evidence."},
                {"role": "user", "content": content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        wire = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(wire, timeout=self.timeout) as response:
            result = json.load(response)
        return json.loads(result["choices"][0]["message"]["content"])

    @staticmethod
    def _json_default(value):
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
