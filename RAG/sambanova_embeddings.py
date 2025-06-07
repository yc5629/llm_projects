from langchain_core.embeddings import Embeddings
from typing import List
import openai

# class SambaNovaEmbeddings(Embeddings):
#     def __init__(self, api_key: str, model: str = "E5-Mistral-7B-Instruct", base_url: str = "https://api.sambanova.ai/v1"):
#         self.client = openai.OpenAI(
#             api_key=api_key,
#             base_url=base_url,
#         )
#         self.model = model

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         embeddings = []
#         for text in texts:
#             response = self.client.embeddings.create(
#                 model=self.model,
#                 input=text
#             )
#             embeddings.append(response.data[0].embedding)
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         response = self.client.embeddings.create(
#             model=self.model,
#             input=text
#         )
#         return response.data[0].embedding

import openai
import time
from typing import List
from langchain_core.embeddings import Embeddings

class SambaNovaEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "E5-Mistral-7B-Instruct",
        base_url: str = "https://api.sambanova.ai/v1",
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents in batches with retry logic."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._retry_request(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query with retry logic."""
        result = self._retry_request([text])
        return result[0]

    def _retry_request(self, input_batch: List[str]) -> List[List[float]]:
        """Handles retry logic for the embedding API call."""
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=input_batch
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                attempt += 1
                print(f"Retry {attempt}/{self.max_retries} after error: {e}")
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay)
