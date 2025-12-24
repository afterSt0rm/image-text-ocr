import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

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
    send_chat_completion_request,
)

app = FastAPI(title="OCR API", description="API for OCR processing of images and PDFs")


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
    Returns validated JSON matching OfferLetterData schema.
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

        # 2. Map Phase: Extract information from each page in parallel
        system_prompt = """You are an OCR assistant specialized in extracting payment and enrollment details from University Offer Letters.
Extract ONLY what you can clearly see on this page. Return ONLY valid JSON matching the schema.

CRITICAL FIELD DEFINITIONS:
- 'course_name': The academic program or language course name (e.g., "Japanese Language Course", "B.Sc Computer Science").
- 'remit_amount': Total amount to be paid/remitted (numeric)
- 'total_tuition_amount': Grand total for the course (includes remit amount, fees, etc.). Numeric float.
- 'student_name': Must be the full name of the student including the first name and lastname/surname.
- 'beneficiary_name': Student's university or college name.
- 'iban': International Bank Account Number. Starts with 2 letters + 2 digits + long alphanumeric. NOTE: Not used in Australia.
- 'swift': 8 or 11 char alphanumeric code (BIC). Required for ALL payments.
- 'bsb': EXACTLY a six-digit code (XXXXXX or XXX-XXX) for Australian banks.
- 'account_number': Bank account of the University. Required if BSB is present.
- 'payment_purpose': The purpose or reference for the payment. IMPORTANT: Look for this on the same page as 'remit_amount' or bank details.
- 'university_address': Full mailing address of the University/College.

RULES:
- For Australian Payments: Expect 'swift' + 'bsb' + 'account_number' (NO IBAN).
- For International Payments: Expect 'swift' + 'iban'.
- Return null for fields not visible on THIS page.
- Do NOT hallucinate. Do not mistake phone numbers for banking details."""

        async def process_page(img_bytes, page_idx):
            img_b64 = bytes_to_base64(img_bytes, max_size=1024)
            return await send_chat_completion_request(
                f"Extract details from Page {page_idx + 1}.",
                images_base64=[img_b64],
                system_prompt=system_prompt,
                response_format=OFFER_LETTER_JSON_SCHEMA,
            )

        map_results_raw = await asyncio.gather(
            *[process_page(img, i) for i, img in enumerate(page_images)]
        )

        # 3. Reduce Phase: Accumulative Consolidation
        consolidation_prompt = f"""You are a JSON consolidation assistant. I have processed an offer letter page by page.
Merge the partial extractions into a single, high-precision JSON object.

Partial Extractions:
{json.dumps(map_results_raw, indent=2)}

REFINEMENT & CONFLICT RESOLUTION RULES:
1. OFFICIAL UNIVERSITY ADDRESS: For 'university_address', identify the address explicitly associated with the 'beneficiary_name' (the University/College). Strictly prioritize the official institution address found on letterheads or footer sections. DO NOT use the student's address or any other unrelated descriptive text simply because it is longer.
2. NAME COMPLETENESS: For 'student_name' and 'beneficiary_name', always prioritize the most complete version of the name (e.g., "University of Sydney" over "Sydney Uni", "John Doe" over "John").
3. BANKING VALIDATION: Strictly enforce the 'Australian vs International' logic.
   - If a valid 6-digit 'bsb' is found, DISCARD any 'iban' found on other pages (likely hallucinations).
   - Ensure 'account_number' is only present if 'bsb' is present.
4. COURSE NAME ACCURACY: Search through all page extractions for a valid academic program name (e.g., "Japanese Language Course") and use that.
5. PURPOSE PROXIMITY: Prioritize the 'payment_purpose' that was extracted from a page containing the 'remit_amount' or banking details.
6. CROSS-PAGE ACCUMULATION: If information for a single field is partial across pages, synthesize the pieces into the most accurate full value.
7. NO HALLUCINATION: If a field is null/missing across all pages, return null.

Return ONLY valid JSON matching the OfferLetterData schema."""

        merge_response = await send_chat_completion_request(
            "Merge the partial JSON extractions provided in the system prompt based on the Refinement Rules.",
            system_prompt=consolidation_prompt,
            response_format=OFFER_LETTER_JSON_SCHEMA,
        )

        # Parse and validate response
        data = json.loads(merge_response)
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
