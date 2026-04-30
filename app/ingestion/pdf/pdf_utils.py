from pdf2image import convert_from_path
from pypdf import PdfReader

def pdf_page_to_image(pdf_path, page_number, dpi=100):
    """
    Converts a specific page of a PDF into an image (1-indexed).
    
    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to convert (1-indexed).
        dpi (int): Resolution of the output image.
        
    Returns:
        PIL.Image.Image: Image object of the PDF page.
    """
    if page_number < 1:
        raise ValueError("Page number must be 1 or greater.")
    
    # Convert the specified PDF page to an image
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
    return pages[0]  # return the single page image

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