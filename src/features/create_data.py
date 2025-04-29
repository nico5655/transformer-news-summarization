# This code was used to create the data

# Run following bash commands before this file
# pip install kaggle
# kaggle datasets download -d sbhatti/news-summarization

import pandas as pd
import os
from tqdm import tqdm
from functions_preprocessing import (
    preprocess_articles,
    preprocess_summaries,
)

def get_allowed_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

cpu_count = get_allowed_cpu_count()
n_process = max(1, cpu_count // 2)

def main():
    # Use already unzipped dataset
    data_path = "news-summarization/data.csv"
    assert os.path.exists(data_path), f"Dataset not found at {data_path}. Make sure you've manually unzipped it."

    # Load and filter dataset
    print("Loading dataset...")
    news_data = pd.read_csv(data_path)

    # Select a small subset for faster preprocessing
    subset_size = 500  # Change this if needed
    if len(news_data) > subset_size:
        print(f"Using a subset of {subset_size} samples out of {len(news_data)}...")
        news_data = news_data.sample(n=subset_size, random_state=42).reset_index(drop=True)

    print("Filtering by article length...")
    lengths_article = news_data["Content"].str.len()
    news_data = news_data[
        (lengths_article >= lengths_article.quantile(0.10))
        & (lengths_article <= lengths_article.quantile(0.90))
    ].reset_index(drop=True)

    # Preprocess articles
    print("Preprocessing articles...")
    content_list = news_data["Content"].tolist()
    news_data.loc[:, "Content"] = list(
        tqdm(
            preprocess_articles(content_list, n_process=n_process, batch_size=32),
            total=len(content_list),
            desc="Articles"
        )
    )

    # Preprocess summaries
    print("Preprocessing summaries...")
    summary_list = news_data["Summary"].tolist()
    news_data.loc[:, "Summary"] = list(
        tqdm(
            preprocess_summaries(summary_list, n_process=n_process, batch_size=32),
            total=len(summary_list),
            desc="Summaries"
        )
    )

    # Save the cleaned data
    print("Saving to Parquet...")
    news_data.to_parquet("news_data_cleaned_subset.parquet", index=False)
    print("Done.")

if __name__ == "__main__":
    main()