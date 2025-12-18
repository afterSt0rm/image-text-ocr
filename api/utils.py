import base64
import io
import os
from typing import List, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

load_dotenv()

# OpenAI client configuration
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-required"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1"),
)


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
            image.save(buffer, format="JPEG", quality=75)
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
) -> str:
    """Send an async chat completion request using OpenAI client."""
    
    content = [{"type": "text", "text": instruction}]
    
    if images_base64:
        for img_b64 in images_base64:
             content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_b64,
                },
            })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        # Build request kwargs
        request_kwargs = {
            "model": "Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            "messages": messages,
            "timeout": 600.0,
        }
        
        # Add response_format if provided (for structured output)
        if response_format:
            request_kwargs["response_format"] = response_format
        
        response = await client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")
