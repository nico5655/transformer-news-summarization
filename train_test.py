import os
import pandas as pd
import evaluate
import argparse
import torch
# import s3fs
from torch import nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer
from src.features.tokenization import parallel_tokenize
from src.models.transformer import Transformer
from src.models.train_models import train_model
from src.evaluation.model_evaluation import generate_summaries_transformer


def get_allowed_cpu_count():
    # Returns the number of CPU cores available for this process.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


cpu_count = get_allowed_cpu_count()
n_process = max(1, cpu_count // 2)
torch.set_num_threads(n_process)


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    parser = argparse.ArgumentParser(description='Training and testing parameters')
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of data to be used for training"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Ratio of training data to be used for validation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Number of training epochs"
    )
    args = parser.parse_args()

    TRAIN_RATIO = args.train_ratio
    BATCH_SIZE = args.batch_size
    VAL_SPLIT = args.val_split
    NUM_EPOCHS = args.num_epochs


    logger.info(f"Using {device} device")

    URL_RAW = "https://minio.lab.sspcloud.fr/arougier/diffusion/news_data_cleaned_share.parquet"

    data_path = os.environ.get("data_path", URL_RAW)
    news_data = pd.read_parquet(data_path)


    data_copy = news_data[:]
    data_copy = news_data.sample(frac=1, random_state=42)

    train_size = int(TRAIN_RATIO * len(data_copy))

    # Slice the dataset
    train_data = data_copy[:train_size]
    test_data = data_copy[train_size:]

    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Test size:  {len(test_data)}")

    logger.info("Tokenizing Content...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized_articles = parallel_tokenize(
        list(train_data["Content"]),
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=512,
    )

    tokenized_articles_test = parallel_tokenize(
        list(test_data["Content"]),
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=512,
    )

    tokenized_summaries = parallel_tokenize(
        list(train_data["Summary"]),
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=129,
    )

    article_ids = tokenized_articles.long()
    summary_ids = tokenized_summaries.long()

    dataset = TensorDataset(tokenized_articles, tokenized_summaries)
    train_dataset, val_dataset = random_split(dataset, [1-VAL_SPLIT, VAL_SPLIT])
    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=n_process, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=n_process
    )

    logger.info('Starting training')

    modelTransformer = Transformer(
        pad_idx=0,
        voc_size=tokenizer.vocab_size,
        hidden_size=128,
        n_head=8,
        max_len=512,
        dec_max_len=512,
        ffn_hidden=128,
        n_layers=3,
    )

    train_model(
        model=modelTransformer,
        dataloader=dataloader,
        val_data_loader=val_dataloader,
        num_epochs=NUM_EPOCHS,
        optimizer=torch.optim.Adam(modelTransformer.parameters(), lr=2e-4),
        loss_fn=nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        ),
        model_name="Transformer",
        device=device,
    )

    modelTransformer.eval()

    rouge = evaluate.load("rouge")

    predictions_transformer = generate_summaries_transformer(
        model=modelTransformer,
        batch_size=BATCH_SIZE,
        tokenized_input=tokenized_articles_test,
        limit=None,
    )

    test_data.loc[:, "predictions_transformer"] = predictions_transformer

    reference_summaries = list(test_data["Summary"])
    results = rouge.compute(
        predictions=predictions_transformer, references=reference_summaries
    )

    logger.info("ROUGE metrics:", results)

if __name__ == "__main__":
    main()