import hashlib
import filetype
from langchain.docstore.document import Document


def generate_hash(data: bytes) -> str:
    hash = hashlib.sha256(data).hexdigest()
    return hash


def get_file_type(file: bytes) -> tuple[str | None, str | None]:
    kind = filetype.guess(file)
    if kind is None:
        return None, None

    file_extension = kind.extension
    mime_type = kind.mime

    return file_extension, mime_type


def calculate_webpage_size(documents: list[Document]) -> int:
    webpage_size = 0
    for doc in documents:
        text = doc.page_content
        webpage_size += len(text.encode("utf-8"))

    return webpage_size
