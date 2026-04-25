"""
Redis caching layer — wraps claim verification results.
Each (claim_text → VerificationResult) is cached for 1 hour.
Cache key is a SHA-256 hash of the claim text (normalised).
"""

import json
import hashlib
import logging
import redis as redis_lib

from app.schemas import VerificationResult
from app.config import settings

logger = logging.getLogger(__name__)


class ClaimCache:
    def __init__(self):
        self.ttl = settings.cache_ttl_seconds
        self._client = None  # lazy connect

    def _get_client(self):
        """Lazy Redis connection — fails gracefully if Redis is down."""
        if self._client is None:
            try:
                self._client = redis_lib.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                self._client.ping()  # test connection
                logger.info("Redis connected.")
            except Exception as e:
                logger.warning(f"Redis unavailable — caching disabled. ({e})")
                self._client = None
        return self._client

    @staticmethod
    def _make_key(claim_text: str) -> str:
        """Normalise claim and hash it into a cache key."""
        normalised = claim_text.lower().strip()
        return "hd:claim:" + hashlib.sha256(normalised.encode()).hexdigest()

    def get(self, claim_text: str) -> VerificationResult | None:
        """Return cached VerificationResult or None if not found / Redis down."""
        client = self._get_client()
        if client is None:
            return None
        try:
            key = self._make_key(claim_text)
            raw = client.get(key)
            if raw:
                logger.info(f"Cache HIT for: {claim_text[:50]}")
                return VerificationResult.model_validate_json(raw)
            logger.info(f"Cache MISS for: {claim_text[:50]}")
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, claim_text: str, result: VerificationResult) -> None:
        """Store a VerificationResult in Redis with TTL."""
        client = self._get_client()
        if client is None:
            return
        try:
            key = self._make_key(claim_text)
            client.setex(key, self.ttl, result.model_dump_json())
            logger.info(f"Cache SET for: {claim_text[:50]}")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def invalidate(self, claim_text: str) -> None:
        """Delete a specific claim from cache."""
        client = self._get_client()
        if client is None:
            return
        try:
            client.delete(self._make_key(claim_text))
        except Exception as e:
            logger.warning(f"Cache invalidate failed: {e}")