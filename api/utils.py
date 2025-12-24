import base64
import io
import os
from typing import List, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

from api.models import NationalIDData, OfferLetterData

load_dotenv()

# OpenAI client configuration
client = AsyncOpenAI(
    api_key=os.getenv("API_KEY", "no-key-required"),
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:8081/v1"),
)


def get_national_id_json_schema() -> dict:
    """Generate JSON schema for structured output using Pydantic's model_json_schema()."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "national_id",
            "strict": True,
            "schema": NationalIDData.model_json_schema(),
        },
    }


def get_offer_letter_json_schema() -> dict:
    """Generate JSON schema for offer letter structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "offer_letter",
            "strict": True,
            "schema": OfferLetterData.model_json_schema(),
        },
    }


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
            image.save(buffer, format="JPEG", quality=100)
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
    """Convert PDF bytes to a list of JPEG image bytes (Legacy)."""
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(pdf_bytes)
    return [pil_to_bytes(img) for img in images]


def render_pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    """Render each page of a PDF to high-quality JPEG bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        # Increase resolution with zoom matrix
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        images.append(img_bytes)

    doc.close()
    return images


def preprocess_image(image_bytes: bytes) -> bytes:
    """Apply denoising and deskewing to image bytes using OpenCV."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return image_bytes

        # 1. Denoising
        # Reduce noise while preserving edges
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # 2. Deskewing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold to get black text on white background
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find all points where pixels are non-zero (text)
        coords = np.column_stack(np.where(gray > 0))
        # Get the minimum area rotated rectangle that contains all points
        angle = cv2.minAreaRect(coords)[-1]

        # The angle returned by minAreaRect is in the range [-90, 0)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If the angle is significant, rotate
        if abs(angle) > 0.5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        # Encode back to bytes
        _, encoded_img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return encoded_img.tobytes()

    except Exception as e:
        print(f"Error during image pre-processing: {e}")
        return image_bytes


def extract_text_and_images_from_pdf(pdf_bytes: bytes) -> List[Tuple[str, List[bytes]]]:
    """
    Extract text and embedded images from PDF bytes.
    Returns a list of tuples: (extracted_text, list_of_image_bytes)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []

    for page in doc:
        # 1. Extract Text
        text = page.get_text()

        # 2. Extract Embedded Images
        image_list = page.get_images()
        page_images = []

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Simple Filter: Ignore small icons/logos (< 150px both dims)
            # We need PIL to check size efficiently
            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size
                if width >= 150 and height >= 150:
                    # Store as tuple (area, bytes) for sorting
                    area = width * height
                    page_images.append((area, image_bytes))
            except Exception:
                continue

        # Sort by area descending (Largest first)
        page_images.sort(key=lambda x: x[0], reverse=True)
        # Return only bytes
        sorted_images = [img[1] for img in page_images]

        results.append((text, sorted_images))

    return results


async def send_chat_completion_request(
    instruction: str,
    images_base64: List[str] = None,
    system_prompt: str = None,
    response_format: dict = None,
    greedy: str = "false",
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
    presence_penalty: float = None,
    max_tokens: int = None,
) -> str:
    """Send an async chat completion request using OpenAI client."""

    content = [{"type": "text", "text": instruction}]

    if images_base64:
        for img_b64 in images_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_b64,
                    },
                }
            )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        # Default generation parameters for QWen3-VL
        if images_base64:
            # For VL
            greedy = greedy if greedy is not None else "false"
            temperature = temperature if temperature is not None else 0.2
            top_p = top_p if top_p is not None else 0.8
            top_k = top_k if top_k is not None else 20
            repetition_penalty = (
                repetition_penalty if repetition_penalty is not None else 1.0
            )
            presence_penalty = presence_penalty if presence_penalty is not None else 1.5
            max_tokens = max_tokens if max_tokens is not None else 16384
        else:
            # For Text
            greedy = greedy if greedy is not None else "false"
            temperature = temperature if temperature is not None else 0.1
            top_p = top_p if top_p is not None else 1.0
            top_k = top_k if top_k is not None else 40
            repetition_penalty = (
                repetition_penalty if repetition_penalty is not None else 1.0
            )
            presence_penalty = presence_penalty if presence_penalty is not None else 2.0
            max_tokens = max_tokens if max_tokens is not None else 32768

        # Build request kwargs
        request_kwargs = {
            "model": "Qwen3-VL-8B-Thinking-GGUF:Q4_K_M",
            "messages": messages,
            "timeout": 600.0,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "extra_body": {
                "greedy": greedy,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            },
        }

        # Add response_format if provided (for structured output)
        if response_format:
            request_kwargs["response_format"] = response_format

        response = await client.chat.completions.create(**request_kwargs)

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")
