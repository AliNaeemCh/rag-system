from pdf2image import convert_from_path
from pypdf import PdfReader
import base64
from io import BytesIO
from enum import Enum

class ImageFormat(str, Enum):
    PIL = "pil"
    DATA_URL = "data_url"

def pil_image_to_data_url(image, format="PNG"):
    """
    Convert PIL image to base64 data URL usable as image_url.
    """
    buffer = BytesIO()
    image.save(buffer, format=format)

    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/{format.lower()};base64,{base64_image}"

def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 100, output_format: ImageFormat = ImageFormat.DATA_URL):
    """
    Converts a specific page of a PDF into an image (1-indexed).
    
    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to convert (1-indexed).
        dpi (int): Resolution of the output image.
    """
    if page_number < 1:
        raise ValueError("Page number must be 1 or greater.")
    
    # Convert the specified PDF page to an image
    page_pil = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)[0]
    if output_format == ImageFormat.DATA_URL:
        return pil_image_to_data_url(page_pil)
    return page_pil

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Returns total number of pages in a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        int: Number of pages in the PDF
    """
    reader = PdfReader(pdf_path)
    return len(reader.pages)

import re

def parse_ranges(range_str: str) -> set[int]:
    """
    Parses range strings like:
        "1-3,5,6"

    Rules:
        - ranges must be ascending (1-3 valid, 3-1 invalid)
        - invalid input raises ValueError
        - whitespace is allowed

    Examples:
        "1-3,5,6" -> {1,2,3,5,6}
        "2,4-6"   -> {2,4,5,6}
    """

    values = set()

    if not range_str or not range_str.strip():
        return values

    for part in range_str.split(","):
        part = part.strip()

        if not part:
            raise ValueError("Empty segment in range string.")

        # Match ranges like 1-3
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)

        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))

            # enforce ascending order
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start must be <= end."
                )

            values.update(range(start, end + 1))
            continue

        # Match single number
        if part.isdigit():
            values.add(int(part))
            continue

        # Anything else is invalid
        raise ValueError(f"Invalid range segment: '{part}'")

    return values