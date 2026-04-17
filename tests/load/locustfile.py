"""
Load test suite for AI Inference Gateway.

Usage:
    locust -f locustfile.py --host=http://localhost:8080

Test scenarios:
    1. ChatCompletionsUser — /v1/chat/completions (OpenAI API)
    2. MessagesUser — /v1/messages (Anthropic API)
    3. HealthCheckUser — /health (monitoring)
    4. ModelsListUser — /v1/models (discovery)

Environment variables:
    GATEWAY_HOST — target host (default: http://localhost:8080)
    API_KEY — optional API key for authentication
"""

import os
import random
from locust import HttpUser, task, between


# Test prompts of varying complexity
SIMPLE_PROMPTS = [
    "What is 2+2?",
    "Hello, how are you?",
    "What's the weather?",
    "Tell me a joke.",
    "What time is it?",
]

CODING_PROMPTS = [
    "Write a Python function that sorts a list using merge sort.",
    "Create a REST API endpoint in FastAPI that accepts JSON.",
    "Implement a binary search tree in TypeScript.",
    "Write a SQL query to find the top 10 customers by revenue.",
    "Create a React component for a searchable dropdown.",
]

REASONING_PROMPTS = [
    "Explain the difference between concurrency and parallelism.",
    "What are the trade-offs between microservices and monoliths?",
    "Compare and contrast REST vs GraphQL APIs.",
    "Explain how TCP congestion control works.",
    "What are the CAP theorem implications for distributed databases?",
]


class ChatCompletionsUser(HttpUser):
    """Simulates OpenAI API chat completion requests."""

    wait_time = between(1, 3)
    weight = 5  # 50% of traffic

    def on_start(self):
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv("API_KEY", "")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @task(3)
    def simple_chat(self):
        """Simple prompt — should route to fast model."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": random.choice(SIMPLE_PROMPTS)}
                ],
                "max_tokens": 100,
                "stream": False,
            },
            headers=self.headers,
            name="/v1/chat/completions [simple]",
        )

    @task(2)
    def coding_chat(self):
        """Coding prompt — should route to coding model."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": random.choice(CODING_PROMPTS)}
                ],
                "max_tokens": 500,
                "stream": False,
            },
            headers=self.headers,
            name="/v1/chat/completions [coding]",
        )

    @task(1)
    def reasoning_chat(self):
        """Reasoning prompt — should route to large model."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": random.choice(REASONING_PROMPTS)}
                ],
                "max_tokens": 500,
                "stream": False,
            },
            headers=self.headers,
            name="/v1/chat/completions [reasoning]",
        )

    @task(1)
    def streaming_chat(self):
        """Streaming request — tests SSE performance."""
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": random.choice(SIMPLE_PROMPTS)}
                ],
                "max_tokens": 100,
                "stream": True,
            },
            headers=self.headers,
            name="/v1/chat/completions [streaming]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # Consume the stream
                chunks = 0
                for line in response.iter_lines():
                    if line:
                        chunks += 1
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class AnthropicMessagesUser(HttpUser):
    """Simulates Anthropic API messages requests."""

    wait_time = between(2, 5)
    weight = 3  # 30% of traffic

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        api_key = os.getenv("API_KEY", "")
        if api_key:
            self.headers["x-api-key"] = api_key

    @task(2)
    def sonnet_request(self):
        """Claude Sonnet — maps to local 9B distill."""
        self.client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": random.choice(CODING_PROMPTS)}
                ],
            },
            headers=self.headers,
            name="/v1/messages [sonnet]",
        )

    @task(1)
    def opus_request(self):
        """Claude Opus — maps to local 35B."""
        self.client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-20250514",
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": random.choice(REASONING_PROMPTS)}
                ],
            },
            headers=self.headers,
            name="/v1/messages [opus]",
        )

    @task(1)
    def haiku_request(self):
        """Claude Haiku — maps to local 0.8B."""
        self.client.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4",
                "max_tokens": 128,
                "messages": [
                    {"role": "user", "content": random.choice(SIMPLE_PROMPTS)}
                ],
            },
            headers=self.headers,
            name="/v1/messages [haiku]",
        )


class HealthCheckUser(HttpUser):
    """Simulates monitoring health checks."""

    wait_time = between(5, 10)
    weight = 1  # 10% of traffic

    @task
    def health(self):
        self.client.get("/health", name="/health")

    @task(3)
    def models(self):
        self.client.get("/v1/models", name="/v1/models")
