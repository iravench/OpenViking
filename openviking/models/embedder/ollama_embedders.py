# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Ollama Embedder Implementation

Uses Ollama's native HTTP API to avoid litellm global state conflicts.
"""

from typing import Any, Dict, List, Optional

import httpx

from openviking.models.embedder.base import DenseEmbedderBase, EmbedResult, truncate_and_normalize


class OllamaDenseEmbedder(DenseEmbedderBase):
    """Ollama Dense Embedder Implementation

    Supports Ollama embedding models via native HTTP API. Ollama is typically run locally
    at http://localhost:11434 and does not require an API key.

    This implementation uses direct HTTP calls to avoid litellm global state conflicts
    with other providers (e.g., when VLM uses a different provider like Moonshot).

    Example:
        >>> embedder = OllamaDenseEmbedder(
        ...     model_name="nomic-embed-text",
        ...     api_base="http://localhost:11434"
        ... )
        >>> result = embedder.embed("Hello world")
        >>> print(len(result.dense_vector))
        768
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimension: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Ollama Dense Embedder

        Args:
            model_name: Ollama model name, defaults to nomic-embed-text
            api_key: API key, optional (not required for local Ollama)
            api_base: API base URL, defaults to http://localhost:11434
            dimension: Dimension, optional
            config: Additional configuration dict
        """
        super().__init__(model_name, config)

        self.api_key = api_key
        self.api_base = api_base or "http://localhost:11434"
        self._dimension = dimension

        # Initialize HTTP client
        client_kwargs: Dict[str, Any] = {
            "base_url": self.api_base,
            "timeout": 60.0,
        }
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
        self._client = httpx.Client(**client_kwargs)

        # Auto-detect dimension if not provided
        if self._dimension is None:
            self._dimension = self._detect_dimension()

    def _detect_dimension(self) -> int:
        """Detect dimension by making an actual API call"""
        try:
            result = self.embed("test")
            return len(result.dense_vector) if result.dense_vector else 768
        except Exception:
            # Common default for Ollama embedding models
            return 768

    def _call_embed_api(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama embed API

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: When API call fails
        """
        try:
            # Ollama supports batch embedding natively
            response = self._client.post(
                "/api/embed",
                json={
                    "model": self.model_name,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns embeddings in data.embeddings
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise RuntimeError("Empty embeddings returned from Ollama")

            return embeddings
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API HTTP error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {str(e)}") from e

    def embed(self, text: str) -> EmbedResult:
        """Perform dense embedding on text

        Args:
            text: Input text

        Returns:
            EmbedResult: Result containing only dense_vector

        Raises:
            RuntimeError: When API call fails
        """
        embeddings = self._call_embed_api([text])
        vector = embeddings[0]

        # Apply dimension truncation if specified
        if self._dimension and len(vector) > self._dimension:
            vector = truncate_and_normalize(vector, self._dimension)

        return EmbedResult(dense_vector=vector)

    def embed_batch(self, texts: List[str]) -> List[EmbedResult]:
        """Batch embedding (Ollama native support)

        Args:
            texts: List of texts

        Returns:
            List[EmbedResult]: List of embedding results

        Raises:
            RuntimeError: When API call fails
        """
        if not texts:
            return []

        embeddings = self._call_embed_api(texts)

        results = []
        for vector in embeddings:
            # Apply dimension truncation if specified
            if self._dimension and len(vector) > self._dimension:
                vector = truncate_and_normalize(vector, self._dimension)
            results.append(EmbedResult(dense_vector=vector))

        return results

    def get_dimension(self) -> int:
        """Get embedding dimension

        Returns:
            int: Vector dimension
        """
        return self._dimension or 768

    def close(self):
        """Release resources (close HTTP client)"""
        if hasattr(self, "_client"):
            self._client.close()
