# This code was used to create the data

# Run following bash commands before this file
# pip install kaggle
# kaggle datasets download -d sbhatti/news-summarization

import pandas as pd
import zipfile
import os
from src.features.functions_preprocessing import (
    preprocess_articles,
    preprocess_summaries,
)


def get_allowed_cpu_count():
    # Returns the number of CPU cores available for this process.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


cpu_count = get_allowed_cpu_count()
n_process = max(1, cpu_count // 2)

with zipfile.ZipFile("news-summarization.zip", "r") as zip_ref:
    zip_ref.extractall("news-summarization")

news_data = pd.read_csv("news-summarization/data.csv")
lengths_article = news_data["Content"].str.len()
lengths_summary = news_data["Summary"].str.len()

news_data = news_data[
    (lengths_article >= lengths_article.quantile(0.10))
    & (lengths_article <= lengths_article.quantile(0.90))
]


news_data.loc[:, "Content"] = preprocess_articles(
    news_data["Content"].tolist(), n_process=n_process, batch_size=32
)
news_data.loc[:, "Summary"] = preprocess_summaries(
    news_data["Summary"].tolist(), n_process=n_process, batch_size=32
)

news_data.to_parquet("news_data_cleaned.parquet", index=False)
