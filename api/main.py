import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

load_dotenv()

from api.models import (
    NationalIDData,
    NationalIDResponse,
    OCRRequest,
    OCRResponse,
    OfferLetterData,
    OfferLetterResponse,
)
from api.utils import (
    bytes_to_base64,
    extract_text_and_images_from_pdf,
    get_national_id_json_schema,
    get_offer_letter_json_schema,
    preprocess_image,
    render_pdf_to_images,
    send_bearer_token_request,
    send_chat_completion_request,
)

# Bearer token API configuration
API_URL = os.getenv("API_URL", "https://gpu-router.server247.info/route/qwen3-vl")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "")

app = FastAPI(title="OCR API", description="API for OCR processing of images and PDFs")


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from a response that may contain markdown code blocks or extra text."""
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(code_block_pattern, response_text)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object pattern { ... }
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, response_text)

    # Try each match, starting with the longest (most complete)
    for match in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # If nothing works, raise an error
    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response", response_text, 0
    )


# Auto-generated JSON Schema for structured output
NATIONAL_ID_JSON_SCHEMA = get_national_id_json_schema()
OFFER_LETTER_JSON_SCHEMA = get_offer_letter_json_schema()


# Helper function to save output
def save_output(content: str, prefix: str = "ocr") -> Path:
    """Save OCR output to markdown file and return the path."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_output_{timestamp}.md"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Saved output to: {output_path}")
    return output_path


@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image_endpoint(
    request_data: OCRRequest = Depends(OCRRequest.as_form_image),
    file: UploadFile = File(...),
):
    """
    OCR endpoint for single images.
    Accepts an image file, text prompt and an optional system prompt.
    """
    try:
        contents = await file.read()

        # Convert to base64
        image_base64 = bytes_to_base64(contents, max_size=2048)

        # Send request to VLM
        response_text = await send_chat_completion_request(
            request_data.prompt,
            images_base64=[image_base64],
            system_prompt=request_data.system_prompt,
        )

        save_output(response_text, prefix="image")

        return OCRResponse(
            filename=file.filename,
            prompt=request_data.prompt,
            system_prompt=request_data.system_prompt,
            response=response_text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf_endpoint(
    request_data: OCRRequest = Depends(OCRRequest.as_form_pdf),
    file: UploadFile = File(...),
):
    """
    OCR endpoint for PDF files with hybrid processing.
    Extracts text and embedded images, then processes each page.
    """
    try:
        contents = await file.read()

        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, detail="This endpoint only accepts PDF files"
            )

        response_texts = []

        # Extract Text + Embedded Images
        pages = extract_text_and_images_from_pdf(contents)

        for i, (extracted_text, page_images) in enumerate(pages):
            images_base64 = []
            for img_bytes in page_images[:1]:
                images_base64.append(bytes_to_base64(img_bytes, max_size=1024))

            # Hybrid Prompt: Digital Text + Figures
            hybrid_prompt = f"""Page {i + 1} Content:
```
{extracted_text[:4000]}
```
I have also attached the {len(images_base64)} key figures/images found on this page.
Please transcribe the full page content in Markdown.
- Use the provided text trace as the source of truth for text.
- For attached images: Provide a comprehensive visual description of the chart/figure, explaining trends or content visible in the image.
- Format code blocks and JSON strictly with correct syntax (```json, ```python).
- Do not hallucinate content not present in the text or images.
{request_data.prompt}"""

            page_response = await send_chat_completion_request(
                hybrid_prompt,
                images_base64=images_base64,
                system_prompt=request_data.system_prompt,
            )
            response_texts.append(f"--- Page {i + 1} ---\n{page_response}")

        final_response = "\n\n".join(response_texts)
        save_output(final_response, prefix="pdf")

        return OCRResponse(
            filename=file.filename,
            prompt=request_data.prompt,
            system_prompt=request_data.system_prompt,
            response=final_response,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/national_id", response_model=NationalIDResponse)
async def ocr_national_id_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all fields from this National ID card."),
):
    """
    OCR endpoint for National ID cards with structured output.
    Returns validated JSON matching NationalIDData schema.
    """
    try:
        contents = await file.read()

        # Convert to base64
        image_base64 = bytes_to_base64(contents, max_size=1024)

        system_prompt = """You are an OCR assistant specialized in extracting information from National ID cards.
Extract all fields from the ID card image and return ONLY valid JSON matching the required schema.
Do not include any explanations or additional text outside the JSON object.

CRITICAL: All dates must be converted to YYYY-MM-DD format regardless of how they appear in the document (e.g., DD-MM-YYYY or MM/DD/YY must be normalized to YYYY-MM-DD)."""

        # Use user's prompt or default
        user_prompt = prompt

        # Send request with structured output
        response_text = await send_chat_completion_request(
            user_prompt,
            images_base64=[image_base64],
            system_prompt=system_prompt,
            response_format=NATIONAL_ID_JSON_SCHEMA,
        )

        # Parse and validate response
        data = json.loads(response_text)
        validated_data = NationalIDData(**data)

        # Save output
        save_output(response_text, prefix="national_id")

        return NationalIDResponse(
            filename=file.filename,
            data=validated_data,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse structured output: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/scanned_pdf", response_model=OCRResponse)
async def ocr_scanned_pdf_endpoint(
    request_data: OCRRequest = Depends(OCRRequest.as_form_image),
    file: UploadFile = File(...),
):
    """
    OCR endpoint for scanned PDFs (PDFs containing only images).
    Renders all pages as images, pre-processes them, and returns full transcription.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, detail="This endpoint only accepts PDF files"
            )

        contents = await file.read()

        # 1. Render PDF pages to images
        page_images = render_pdf_to_images(contents)

        if not page_images:
            raise HTTPException(
                status_code=400, detail="Could not extract images from PDF"
            )

        response_texts = []

        # 2. Process each page
        for i, img_bytes in enumerate(page_images):
            # Apply OpenCV pre-processing (denoising, deskewing)
            # preprocessed = preprocess_image(img_bytes)
            # Convert to base64
            image_base64 = bytes_to_base64(img_bytes, max_size=1024)

            # Send request for this page
            page_response = await send_chat_completion_request(
                f"Transcribe Page {i + 1} accurately in Markdown.",
                images_base64=[image_base64],
                system_prompt=request_data.system_prompt,
            )
            response_texts.append(f"--- Page {i + 1} ---\n{page_response}")

        final_response = "\n\n".join(response_texts)
        save_output(final_response, prefix="scanned_pdf")

        return OCRResponse(
            filename=file.filename,
            prompt=request_data.prompt,
            system_prompt=request_data.system_prompt,
            response=final_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/offer_letter", response_model=OfferLetterResponse)
async def ocr_offer_letter_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form("Extract structured information from this offer letter. "),
):
    """
    OCR endpoint for Offer Letters with multi-page support and pre-processing.
    Uses Bearer token authentication API for chat completion requests.
    Returns validated JSON matching OfferLetterData schema.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, detail="This endpoint only accepts PDF files"
            )

        if not BEARER_TOKEN:
            raise HTTPException(
                status_code=500,
                detail="BEARER_TOKEN environment variable is not configured",
            )

        contents = await file.read()

        # 1. Render PDF pages to images
        page_images = render_pdf_to_images(contents)

        if not page_images:
            raise HTTPException(
                status_code=400, detail="Could not extract images from PDF"
            )

        # 2. Map Phase: Extract information from each page in parallel
        system_prompt = """You are an OCR assistant specialized in extracting payment and enrollment details from University Offer Letters.
Extract ONLY what you can clearly see on this page. Return ONLY valid JSON.

CRITICAL FIELD DEFINITIONS:
- 'course_name': The academic program or language course name (e.g., "Japanese Language Course", "B.Sc Computer Science"). String.
- 'remit_amount': Total amount to be paid/remitted. Numeric float.
- 'total_tuition_amount': Grand total for the course (includes remit amount, fees, etc.). Numeric float.
- 'remit_currency': Valid currency code for the payment (e.g., "USD", "AUD", "JPY"). String.
- 'student_name': Must be the full name of the student including the first name and lastname/surname. String.
- 'beneficiary_name': Student's university or college name. String.
- 'iban': International Bank Account Number. Starts with 2 letters + 2 digits + long alphanumeric. NOTE: Not used in Australia. String or null.
- 'swift': 8 or 11 char alphanumeric code (BIC). Required for ALL payments. String or null.
- 'bsb': EXACTLY a six-digit code (XXXXXX or XXX-XXX) for Australian banks. String or null.
- 'account_number': Bank account of the University. Required if BSB is present. String or null.
- 'payment_purpose': The purpose or reference for the payment. IMPORTANT: Look for this on the same page as 'remit_amount' or bank details. String or null.
- 'university_address': Full mailing address of the University/College. String.

RULES:
- For Australian Payments: Expect 'swift' + 'bsb' + 'account_number' (NO IBAN).
- For International Payments: Expect 'swift' + 'iban'.
- Return null for fields not visible on THIS page.
- Do NOT hallucinate. Do not mistake phone numbers for banking details.

OUTPUT FORMAT:
Return ONLY a valid JSON object with these exact keys:
{
  "course_name": "string or null",
  "total_tuition_amount": 0.0,
  "remit_amount": 0.0,
  "remit_currency": "string or null",
  "student_name": "string or null",
  "beneficiary_name": "string or null",
  "iban": "string or null",
  "swift": "string or null",
  "bsb": "string or null",
  "payment_purpose": "string or null",
  "account_number": "string or null",
  "university_address": "string or null"
}"""

        async def process_page(img_bytes: bytes, page_idx: int) -> str:
            return await send_bearer_token_request(
                api_url=API_URL,
                bearer_token=BEARER_TOKEN,
                prompt=f"Extract details from Page {page_idx + 1}. Return ONLY valid JSON.",
                system_prompt=system_prompt,
                image=img_bytes,
                image_filename=f"page_{page_idx + 1}.jpg",
            )

        map_results_raw = await asyncio.gather(
            *[process_page(img, i) for i, img in enumerate(page_images)]
        )

        # 3. Parse each page result to clean JSON
        parsed_pages = []
        for i, raw_result in enumerate(map_results_raw):
            try:
                parsed = extract_json_from_response(raw_result)
                # Remove null/None values to reduce size
                cleaned = {
                    k: v for k, v in parsed.items() if v is not None and v != "null"
                }
                if cleaned:
                    parsed_pages.append(cleaned)
            except Exception:
                continue  # Skip unparseable pages

        if not parsed_pages:
            raise HTTPException(
                status_code=500, detail="Could not extract any data from the PDF pages"
            )

        # 4. Incremental Merge: Merge pages two at a time
        async def merge_two_extractions(extraction1: dict, extraction2: dict) -> dict:
            """Merge two JSON extractions into one."""
            merge_prompt = f"""Merge these two JSON extractions from an offer letter into one combined JSON.

Extraction 1:
{json.dumps(extraction1)}

Extraction 2:
{json.dumps(extraction2)}

REFINEMENT & CONFLICT RESOLUTION RULES:
1. OFFICIAL UNIVERSITY ADDRESS: For 'university_address', identify the address explicitly associated with the 'beneficiary_name' (the University/College). DO NOT use the student's address or any other unrelated descriptive text simply because it is longer.
2. OFFICIAL UNIVERSITY NAME: For 'beneficiary_name', identify the official University/College name the student is going to study. Strictly prioritize the official institution name found on letterheads. DO NOT use the bank's name or any other unrelated company name simply because it is longer
2. NAME COMPLETENESS: For 'student_name' and 'beneficiary_name', always prioritize the most complete version of the name (e.g., "University of Sydney" over "Sydney Uni", "John Doe" over "John").
3. BANKING VALIDATION: Strictly enforce the 'Australian vs International' logic.
   - If a valid 6-digit 'bsb' is found, DISCARD any 'iban' found on other pages (likely hallucinations).
   - Ensure 'account_number' is only present if 'bsb' is present.
4. COURSE NAME ACCURACY: Search through all page extractions for a valid academic program name (e.g., "Japanese Language Course") and use that.
5. PURPOSE PROXIMITY: Prioritize the 'payment_purpose' that was extracted from a page containing the 'remit_amount' or banking details (for example, tutition fee payment, admission fee deposit, living expenses, etc).
6. CROSS-PAGE ACCUMULATION: If information for a single field is partial across pages, synthesize the pieces into the most accurate full value.
7. PAYMENT VALIDATION
    - 'remit_amount': Total amount to be paid/remitted. Numeric float. In most cases remit amount is lower than total tuition amount.
    - 'total_tuition_amount': Grand total for the course (includes remit amount, fees, etc.). Numeric float
9. NO HALLUCINATION: If a field is null/missing across all pages, use an null for required string fields and 0.0 for required numeric fields.

Return ONLY valid JSON with keys: course_name, total_tuition_amount, remit_amount, remit_currency, student_name, beneficiary_name, iban, swift, bsb, payment_purpose, account_number, university_address."""

            response = await send_bearer_token_request(
                api_url=API_URL,
                bearer_token=BEARER_TOKEN,
                prompt="Merge these two extractions. Return ONLY valid JSON.",
                system_prompt=merge_prompt,
            )
            return extract_json_from_response(response)

        # If only one page, use it directly
        if len(parsed_pages) == 1:
            merged_result = parsed_pages[0]
        else:
            # Incrementally merge: start with first, merge with each subsequent
            merged_result = parsed_pages[0]
            for i in range(1, len(parsed_pages)):
                merged_result = await merge_two_extractions(
                    merged_result, parsed_pages[i]
                )

        # Ensure all required fields exist with defaults
        default_values = {
            "course_name": "",
            "total_tuition_amount": 0.0,
            "remit_amount": 0.0,
            "remit_currency": "",
            "student_name": "",
            "beneficiary_name": "",
            "iban": None,
            "swift": None,
            "bsb": None,
            "payment_purpose": None,
            "account_number": None,
            "university_address": "",
        }
        for key, default in default_values.items():
            if key not in merged_result or merged_result[key] is None:
                if default is not None:  # Only set default for required fields
                    merged_result[key] = default

        merge_response = json.dumps(merged_result)

        # Parse and validate response
        data = extract_json_from_response(merge_response)
        validated_data = OfferLetterData(**data)

        # Save output
        save_output(merge_response, prefix="offer_letter")

        return OfferLetterResponse(
            filename=file.filename,
            data=validated_data,
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse structured output: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
