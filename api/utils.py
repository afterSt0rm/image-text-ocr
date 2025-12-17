import base64
import io
import os
from typing import List

import requests
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from PIL import Image

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8081")


def bytes_to_base64(image_bytes: bytes, max_size: int = 1024) -> str:
    """Convert image bytes to base64 string, resizing if necessary."""
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Resize if larger than max_size
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))

        # Convert to JPEG bytes
        with io.BytesIO() as buffer:
            image = image.convert("RGB")  # Ensure RGB for JPEG
            image.save(buffer, format="JPEG", quality=85)
            resized_bytes = buffer.getvalue()

        return (
            f"data:image/jpeg;base64,{base64.b64encode(resized_bytes).decode('utf-8')}"
        )
    except Exception as e:
        # Fallback for non-image bytes or errors
        print(f"Error resizing image: {e}")
        return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def pil_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to bytes."""
    with io.BytesIO() as bio:
        image.save(bio, format="JPEG")
        return bio.getvalue()


def convert_pdf_to_images(pdf_bytes: bytes) -> List[bytes]:
    """Convert PDF bytes to a list of JPEG image bytes."""
    images = convert_from_bytes(pdf_bytes)
    return [pil_to_bytes(img) for img in images]


def send_chat_completion_request(
    instruction: str,
    image_base64_url: str,
    base_url: str = BASE_URL,
    system_prompt: str = None,
) -> str:
    """Send a chat completion request to the VLM."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64_url,
                    },
                },
            ],
        }
    ]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "max_tokens": 1000,
            "messages": messages,
        },
        timeout=60,  # Add timeout to prevent hanging connections
    )

    if not response.ok:
        raise Exception(f"Server error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]
