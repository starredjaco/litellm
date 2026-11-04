from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.caching.redis_cache import RedisCache
from litellm.rust_bridge._native import NativeRedisClient

_KWARGS: Final = TypeAdapter(dict[str, object])


def _redis_kwargs(connection: RedisCache) -> Mapping[str, object]:
    return _KWARGS.validate_python(
        connection.redis_kwargs  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped legacy dict
    )


def native_client(connection: RedisCache | None) -> NativeRedisClient | None:
    if connection is None:
        return None
    return NativeRedisClient.connect(_redis_kwargs(connection), namespace=connection.namespace)
