import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def preprocess_articles(texts, n_process, batch_size=32):
    """
    Args:
        texts (List[str]): List of text documents to preprocess.
        batch_size (int): Batch size for parallel processing.
        n_process (int): Number of processes for parallel execution.

    Returns:
        List[str]: A list of cleaned articles (lemmatized, no stopwords/punct/spaces).
    """
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts


def preprocess_summaries(texts, n_process, batch_size=32):
    """
    Performs minimal preprocessing for summaries (lowercasing + tokens).

    Args:
        texts (List[str]): List of summary texts.
        batch_size (int): Batch size for parallel processing.
        n_process (int): Number of processes for parallel execution.

    Returns:
        List[str]: A list of preprocessed summaries.
    """
    cleaned_summaries = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [token.text.lower() for token in doc if not token.is_space]
        cleaned_summaries.append(" ".join(tokens))
    return cleaned_summaries
