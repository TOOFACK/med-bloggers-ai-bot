import mimetypes
import os
from typing import Optional, Tuple
from uuid import uuid4

import aioboto3

from config import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_MEDIA_PREFIX,
    S3_PUBLIC_BASE_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)


class S3ConfigError(RuntimeError):
    pass


def _require_bucket() -> str:
    if not S3_BUCKET:
        raise S3ConfigError("S3_BUCKET is not configured")
    return S3_BUCKET


def _client_kwargs() -> dict:
    if not S3_ACCESS_KEY_ID or not S3_SECRET_ACCESS_KEY:
        raise S3ConfigError("S3 credentials are not configured")

    kwargs = {
        "aws_access_key_id": S3_ACCESS_KEY_ID,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY,
    }
    if S3_REGION:
        kwargs["region_name"] = S3_REGION
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return kwargs


def _build_object_key(filename: str, user_id: str | int) -> str:
    ext = os.path.splitext(filename)[1] or ".jpg"
    unique = uuid4().hex
    prefix = S3_MEDIA_PREFIX.rstrip("/")
    return f"{prefix}/{user_id}/{unique}{ext}"


def build_public_url(object_key: str) -> str:
    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{object_key}"
    bucket = _require_bucket()
    if S3_ENDPOINT_URL:
        base = S3_ENDPOINT_URL.rstrip("/")
        return f"{base}/{bucket}/{object_key}"
    region_part = f".{S3_REGION}" if S3_REGION else ""
    return f"https://{bucket}.s3{region_part}.amazonaws.com/{object_key}"


async def upload_bytes(
    data: bytes, filename: str, user_id: str | int
) -> Tuple[str, str]:
    bucket = _require_bucket()
    object_key = _build_object_key(filename, user_id)
    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "image/jpeg"

    session = aioboto3.Session()
    async with session.client("s3", **_client_kwargs()) as client:
        await client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
    return object_key, build_public_url(object_key)


async def delete_object(object_key: Optional[str]) -> None:
    if not object_key:
        return
    bucket = _require_bucket()
    session = aioboto3.Session()
    async with session.client("s3", **_client_kwargs()) as client:
        await client.delete_object(Bucket=bucket, Key=object_key)
