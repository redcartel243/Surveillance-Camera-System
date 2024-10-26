import redis
import numpy as np
import json

class FaceCache:
    def __init__(self, max_size=100, redis_host='127.0.0.1', redis_port=6379, redis_db=0):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.max_size = max_size

    def _encoding_to_str(self, encoding):
        """Convert the numpy array encoding to a string for Redis storage."""
        return json.dumps(encoding.tolist())

    def _str_to_encoding(self, encoding_str):
        """Convert stored string back to numpy array."""
        return np.array(json.loads(encoding_str))

    def get(self, face_encoding):
        """Retrieve face name if it exists in Redis."""
        encoding_str = self._encoding_to_str(face_encoding)
        return self.redis_client.get(encoding_str)

    def add(self, face_encoding, name):
        """Add a face encoding and its name to the Redis cache."""
        encoding_str = self._encoding_to_str(face_encoding)
        if self.redis_client.dbsize() >= self.max_size:
            oldest_key = self.redis_client.randomkey()
            self.redis_client.delete(oldest_key)
        self.redis_client.set(encoding_str, name)
