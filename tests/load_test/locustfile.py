import time
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    # Wait between 1 and 3 seconds between requests
    wait_time = between(1, 3)

    @task
    def generate_text(self):
        prompt = "Explain the concept of Tensor Parallelism in deep learning."
        
        payload = {
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0.7,
            "stream": True
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "sk-faang-resume-project-12345"
        }
        
        start_time = time.time()
        
        # We use SSE streaming endpoint
        with self.client.post("/generate", json=payload, headers=headers, stream=True, catch_response=True) as response:
            if response.status_code == 429:
                response.success() # Rate limit triggered correctly
                return
            elif response.status_code != 200:
                response.failure(f"Failed! Status code: {response.status_code}")
                return
                
            first_token_received = False
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        if not first_token_received:
                            # TTFT (Time To First Token) successfully received
                            ttft = time.time() - start_time
                            first_token_received = True
