import os
import time
import torch
from loguru import logger


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0

    # train step
    for step, batch in enumerate(dataloader):
        input_batch, summary_batch = batch
        input_batch = input_batch.to(device)
        summary_batch = summary_batch.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_batch.long(), summary_batch[:, :-1])
        shifted_target = summary_batch[:, 1:]  # Shift target for loss
        loss = loss_fn(
            outputs.reshape(-1, outputs.shape[-1]), shifted_target.reshape(-1)
        )

        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if step % 1000 == 0:
            logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def val_epoch(model, dataloader, device, loss_fn):
    model.eval()
    total_val_loss = 0
    for step, batch in enumerate(dataloader):
        input_batch, summary_batch = batch
        input_batch = input_batch.to(device)
        summary_batch = summary_batch.to(device)

        # Forward pass
        outputs = model(input_batch.long(), summary_batch[:, :-1])
        shifted_target = summary_batch[:, 1:]  # Shift target for loss
        loss = loss_fn(
            outputs.reshape(-1, outputs.shape[-1]), shifted_target.reshape(-1)
        )

        total_val_loss += loss.item()

    # Calculate average loss for the epoch
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss


def train_model(
    model,
    dataloader,
    val_data_loader,
    num_epochs,
    optimizer,
    loss_fn,
    model_name,
    device
):
    """
    Generalized function to train a model.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - dataloader (DataLoader): DataLoader for the training data.
    - num_epochs (int): Number of epochs to train for.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - loss_fn (torch.nn.Module): Loss function.
    - model_name (str): Name of the model for saving checkpoints.
    - device (torch.device): Device to train on (CPU or GPU).

    Returns:
    - None
    """
    model.to(device)
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch)
        avg_val_loss = val_epoch(model, val_data_loader, device, loss_fn)

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Average Loss: {avg_loss:.4f} - "
            f"Average Val Loss: {avg_val_loss:.4f} - "
            f"Time: {epoch_duration:.2f}s"
        )

        os.makedirs("model_weights", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"model_weights/{model_name.lower()}_weights_{epoch+1}_epochs.pth",
        )

    # Measure total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logger.info(f"Total training time: {total_training_time:.2f}s")