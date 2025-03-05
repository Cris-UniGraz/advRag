from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_splitter(tokenizer_name, chunk_size: int, chunk_overlap: int):
    """
    Initializes and returns a text splitter configured to split documents
    into chunks of up to `chunk_size` tokens using the specified tokenizer.

    Parameters:
    - chunk_size: The maximum number of tokens for each chunk.
    - tokenizer_name: The name of the tokenizer to use. It can be "openai" or the name of a Hugging Face tokenizer.

    Returns:
    - A configured text splitter object (`RecursiveCharacterTextSplitter`) that can be used to split documents into chunks.
    """

    if tokenizer_name == "openai":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            model_name="gpt-4",
            is_separator_regex=False,
            add_start_index=True,
            strip_whitespace=True,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )

    return text_splitter
