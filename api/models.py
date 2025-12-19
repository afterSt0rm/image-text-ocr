from datetime import date
from typing import Optional

from fastapi import Form
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image to do requirements:
- Recognize and Extract all visible text in the image as accurately as possible without any additional explanations or comments.
- Pay close attention to maintaining the original hierarchy and formatting, including any headings, subheadings, lists, tables or inline text.
- If any text elements are ambiguous or partially readable, include them with appropriate notes or markers, such as [illegible].
- Preserve the spatial relationships where applicable by mimiching the document layout in Markdown.
- Don't omit any part of the page including headers, footers, tables, and subtext.
Provide only the transcription without any additional comments."""


class OCRRequest(BaseModel):
    prompt: str = "What is in this image?"
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def as_form_image(
        cls,
        prompt: str = Form("What is in this image?"),
        system_prompt: Optional[str] = Form(DEFAULT_SYSTEM_PROMPT),
    ):
        """Form method for image endpoint using DEFAULT_SYSTEM_PROMPT."""
        return cls(prompt=prompt, system_prompt=system_prompt)

    @classmethod
    def as_form_pdf(
        cls,
        prompt: str = Form("Note down the information from the given document?"),
        system_prompt: Optional[str] = Form(
            "You are an advanced OCR assistant capable of reading dense text and multimodal documents. "
            "Your output must be strictly structured Markdown. "
            "Rules:\n"
            "1. Use proper headings (##) for structure.\n"
            "2. Ensure that the extracted text is organized and presented in a structured Markdown format.\n"
            "3. Pay close attention to maintaining the original hierarchy and formatting, includeing any headings, subheadings, lists, tables or inline text.\n"
            "4. If any text elements are ambiguous or partially readable, include them with appropriate notes or markers, such as [illegible].\n"
            "5. Preserve the spatial relationships where applicable by mimicking the document layout in Markdown.\n"
            "6. Don't omit any part of the page including headers, footers, tables, and subtext.\n"
            "7. Enclose ALL code snippets in correct triple-backtick code blocks with the language specified (e.g., ```python, ```json, ```bash).\n"
            "8. Format all tables using Markdown table syntax.\n"
            "9. Use LaTeX for math formulas, enclosed in $...$ for inline and $$...$$ for block equations.\n"
            "10. Do not output conversational filler."
        ),
    ):
        """Form method for PDF endpoint with hybrid processing prompt."""
        return cls(prompt=prompt, system_prompt=system_prompt)


class OCRResponse(BaseModel):
    filename: str
    prompt: str
    system_prompt: Optional[str]
    response: str


class NationalIDData(BaseModel):
    """Structured data extracted from a National ID card."""

    nationality: str
    sex: str
    surname: str
    given_name: str
    mother_name: str
    father_name: str
    date_of_birth: date = Field(..., description="Date of birth in YYYY-MM-DD format")
    date_of_issue: date = Field(..., description="Date of issue in YYYY-MM-DD format")
    national_id_number: str


class NationalIDResponse(BaseModel):
    """Response model for National ID extraction endpoint."""

    filename: str
    data: NationalIDData


class OfferLetterData(BaseModel):
    """Structured data extracted from an offer letter."""

    course_name: str = Field(..., description="Name of the course/program")
    remit_amount: float = Field(..., description="Amount to be remitted/paid")
    remit_currency: str = Field(..., description="Currency of the remit amount")
    student_name: str = Field(..., description="Name of the student")
    beneficiary_name: str = Field(
        ..., description="Name of the university or college (beneficiary)"
    )
    iban: Optional[str] = Field(None, description="IBAN code for the payment")
    swift: Optional[str] = Field(None, description="SWIFT code for the payment")
    university_address: str = Field(
        ..., description="Full address of the university/beneficiary"
    )


class OfferLetterResponse(BaseModel):
    """Response model for offer letter extraction endpoint."""

    filename: str
    data: OfferLetterData
