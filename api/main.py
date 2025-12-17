import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from api.models import OCRRequest, OCRResponse
from api.utils import (
    bytes_to_base64,
    convert_pdf_to_images,
    send_chat_completion_request,
)

app = FastAPI()


@app.get("/healthcheck")
async def healthcheck():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    request_data: OCRRequest = Depends(OCRRequest.as_form),
    file: UploadFile = File(...),
):
    """
    Endpoint to perform OCR on an uploaded image or PDF.
    Accepts an image/PDF file, text prompt and an optional system prompt.
    """
    try:
        # Read file contents
        contents = await file.read()
        
        response_texts = []

        if file.content_type == "application/pdf":
            # Convert PDF to images
            images_bytes = convert_pdf_to_images(contents)
            
            for i, img_bytes in enumerate(images_bytes):
                image_base64 = bytes_to_base64(img_bytes)
                page_response = send_chat_completion_request(
                    request_data.prompt,
                    image_base64,
                    system_prompt=request_data.system_prompt,
                )
                response_texts.append(f"--- Page {i+1} ---\n{page_response}")
        else:
            # Assume Image
            # Convert to base64
            image_base64 = bytes_to_base64(contents)

            # Send request to VLM
            response_text = send_chat_completion_request(
                request_data.prompt,
                image_base64,
                system_prompt=request_data.system_prompt,
            )
            response_texts.append(response_text)

        final_response = "\n\n".join(response_texts)

        return OCRResponse(
            filename=file.filename,
            prompt=request_data.prompt,
            system_prompt=request_data.system_prompt,
            response=final_response,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
