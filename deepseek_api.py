import requests
import time
import logging

class DeepSeekAPI:
    def __init__(self, api_key, max_retries=5, retry_backoff=2):
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt, temperature=0.7, max_tokens=512):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    logging.warning("Rate limit hit. Retrying...")
                    time.sleep(self.retry_backoff * attempt)
                else:
                    logging.error(f"API error: {response.status_code} - {response.text}")
                    time.sleep(self.retry_backoff * attempt)
            except Exception as e:
                logging.error(f"Request failed: {e}")
                time.sleep(self.retry_backoff * attempt)
        raise RuntimeError("DeepSeek API failed after retries.")
