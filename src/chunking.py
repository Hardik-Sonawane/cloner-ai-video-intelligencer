try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(segments):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for segment in segments:

        sub_chunks = splitter.split_text(segment["text"])

        for chunk in sub_chunks:
            chunks.append(
                {
                    "text": chunk,
                    "start": segment["start"],
                    "end": segment["end"]
                }
            )

    return chunks
