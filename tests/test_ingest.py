import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image

from vision_rag.ingest.document_loader import DocumentLoader

@pytest.fixture
def mock_pdf_path(tmp_path):
    """Fixture to create a temporary dummy PDF file."""
    pdf_file = tmp_path / "dummy.pdf"
    pdf_file.write_text("dummy pdf content")
    return pdf_file

@patch("vision_rag.ingest.document_loader.convert_from_path")
def test_document_loader_success(mock_convert, mock_pdf_path):
    # Setup mock to return two dummy PIL Images
    mock_image1 = MagicMock(spec=Image.Image)
    mock_image2 = MagicMock(spec=Image.Image)
    mock_convert.return_value = [mock_image1, mock_image2]
    
    loader = DocumentLoader(dpi=200, fmt="png")
    images = loader.load_pdf(mock_pdf_path)
    
    # Verify conversion
    mock_convert.assert_called_once_with(
        pdf_path=str(mock_pdf_path),
        dpi=200,
        fmt="png",
        thread_count=4
    )
    assert len(images) == 2
    assert images[0] == mock_image1
    assert images[1] == mock_image2

def test_document_loader_file_not_found():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf("non_existent_file.pdf")
