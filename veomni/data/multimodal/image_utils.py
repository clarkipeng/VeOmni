import io
import random
import math
from io import BytesIO
from typing import ByteString, List, Union

import numpy as np
import requests
from PIL import Image


ImageInput = Union[
    Image.Image,
    np.ndarray,
    ByteString,
    str,
]


def load_image_bytes_from_path(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return image_bytes.getvalue()


def save_image_bytes_to_file(image_bytes, output_path):
    image_bytes = io.BytesIO(image_bytes)
    image = Image.open(image_bytes).convert("RGB")
    image.save(output_path)


def smart_resize(
    image: Image.Image,
    scale_factor: int = None,
    image_min_pixels: int = None,
    image_max_pixels: int = None,
    max_ratio: int = None,
    **kwargs,
):
    width, height = image.size
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if image_max_pixels is not None and h_bar * w_bar > image_max_pixels:
        beta = math.sqrt((height * width) / image_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)
    if image_min_pixels is not None and h_bar * w_bar < image_min_pixels:
        beta = math.sqrt(image_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)
    image = image.resize((w_bar, h_bar))
    return image


def load_image_from_path(image: str, **kwargs):
    import os
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("s3://"):
        # Handle S3 paths using boto3 (like BlobLearn does)
        try:
            import boto3
            # Parse s3://bucket/key
            parts = image[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response["Body"].read()
            image_obj = Image.open(BytesIO(image_bytes))
        except ImportError:
            raise ImportError("boto3 is required for S3 image paths. Install it with: pip install boto3")
    else:
        # Check if it's a relative path that should be resolved to S3
        # If image_bucket is provided in kwargs, construct S3 path
        image_bucket = kwargs.get("image_bucket")
        if image_bucket and not os.path.isabs(image) and not os.path.exists(image):
            # Construct S3 path: s3://bucket/path
            s3_path = f"s3://{image_bucket}/{image}"
            try:
                import boto3
                # Parse s3://bucket/key
                parts = s3_path[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""
                s3_client = boto3.client("s3")
                response = s3_client.get_object(Bucket=bucket, Key=key)
                image_bytes = response["Body"].read()
                image_obj = Image.open(BytesIO(image_bytes))
            except ImportError:
                raise ImportError("boto3 is required for S3 image paths. Install it with: pip install boto3")
            except Exception as e:
                # If image doesn't exist in S3, raise a more informative error
                raise FileNotFoundError(f"Image not found at S3 path: {s3_path}. Original error: {e}")
        else:
            image_obj = Image.open(image)
    return image_obj.convert("RGB")


def load_image_from_bytes(image: bytes, **kwargs):
    return Image.open(BytesIO(image)).convert("RGB")


def load_image(image: ImageInput, **kwargs):
    if isinstance(image, str):
        return load_image_from_path(image, **kwargs)
    elif isinstance(image, bytes):
        return load_image_from_bytes(image, **kwargs)
    else:
        raise NotImplementedError


def random_crop(image: Image.Image, crop_ratio: float = 0.6, **kwargs):
    width, height = image.size
    
    # Calculate crop bounds based on crop_ratio
    # If crop_ratio is 0.6, we keep at least 60% of the image.
    # The "discardable" margin is (1 - crop_ratio) / 2 on each side.
    margin = (1 - crop_ratio) / 2
    crop_min = margin
    crop_max = 1 - margin
    
    # Crop boundaries:
    # left: 0 to crop_min * width
    # right: crop_max * width to width
    # top: 0 to crop_min * height
    # bottom: crop_max * height to height
    
    left = random.randint(0, int(crop_min * width))
    right = random.randint(int(crop_max * width), width)
    top = random.randint(0, int(crop_min * height))
    bottom = random.randint(int(crop_max * height), height)
    
    # Ensure valid crop (right > left, bottom > top) - guaranteed by logic above if crop_max > crop_min
    image = image.crop((left, top, right, bottom))
    return image


def random_resize(image: Image.Image, min_ratio=0.5, max_ratio=1.5):
    ratio = random.uniform(min_ratio, max_ratio)
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)
    return image.resize((new_width, new_height))


def fetch_images(
    images: List[ImageInput],
    do_random_crop: bool = False,
    crop_ratio: float = 0.6,
    resize_ratio: float = 0.5,
    **kwargs
):
    images = [load_image(image, **kwargs) for image in images]
    max_image_nums = kwargs.get("max_image_nums", len(images))
    images = images[:max_image_nums]
    
    if do_random_crop:
        images = [random_crop(image, crop_ratio=crop_ratio, **kwargs) for image in images]
        
        # resize_ratio defines the minimum scale. Max scale is symmetric around 1.0?
        # Or simply [resize_ratio, 2 - resize_ratio] as discussed.
        # If resize_ratio is 0.5, range is [0.5, 1.5].
        min_ratio = resize_ratio
        max_ratio = 2 - resize_ratio
        
        images = [random_resize(image, min_ratio=min_ratio, max_ratio=max_ratio) for image in images]
    
    images = [smart_resize(image, **kwargs) for image in images]
    return images
