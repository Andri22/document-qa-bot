def format_docs(docs: list) -> str:
    """Convert retrieved docs to single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list) -> list[str]:
    """Extract unique source filenames from docs."""
    return list(set([doc.metadata.get("source", "unknown") for doc in docs]))


def validate_pdf(filename: str) -> bool:
    """Check if uploaded file is a PDF."""
    return filename.lower().endswith(".pdf")
