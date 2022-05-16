from copy import deepcopy
import logging
import datetime
import argparse
import random
import numpy as np
import json
from typing import *
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from datatuner.lm.custom import datatuner_dataset
from nltk.translate.bleu_score import sentence_bleu

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)

# Configuration details. These could be passed as command line arguments but are done this way
# for simplicity.

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--base_dataset_path", type=str, help="Path of the dataset.")
    parser.add_argument("--task_config_path", type=str, help="Path to the tokenization config file")
    parser.add_argument("--special_tokens_path", type=str, default=None, help="Path to the special tokens file")
    parser.add_argument("--train_params_path", type=str, help="JSON file with training parameters.")
    parser.add_argument("--save_dir_path", type=str, default="./save", help="Path to the save directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Short name of the model")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    # parser.add_argument("--patience", type=int, default=1, help="patience parameter for early stopping")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward(model, batch, device: str, pad_token_id: int = 0):
    """When we call model() with labels, they will be:
    - automatically right shifted by 1 (for teacher forcing)
    - prepended by BOS=Beginning of sequence which is a PAD token
    - any token that was -100 will be masked_fill_ to <pad> for teacher forcing

    Args:
        model (_type_)
        device (str)
        batch (_type_)
        pad_token_id (int): defaults to 0

    Returns:
        float: loss value
    """
    source_ids = batch["source_input_ids"].to(device, dtype=torch.long)
    source_mask = batch["source_attention_mask"].to(device, dtype=torch.long)
    target_ids = batch["target_input_ids"].to(device, dtype=torch.long)
    #* padded ids are set to -100, so that they are ignored during loss calculation
    target_ids[target_ids[: ,:] == pad_token_id] = -100
    label_ids = target_ids.to(device)
    out_dict = model(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
    loss = out_dict[0]
    return loss


def main():
    #* set seed and args
    log.info("Parsing arguments")
    args = parse_arguments()
    set_seed(args.seed)

    #* check if there are any additional training args and add them to args
    log.info(f"Checking additional parameters at {args.train_params_path}")
    if args.train_params_path:
        train_params = json.load(open(args.train_params_path, "r"))
        for param in train_params:
            if not hasattr(args, param):
                setattr(args, param, train_params[param])

    #* load model and tokenizer
    log.info(f"Loading model and tokenizer")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    #* check and eventually update special_tokens_path
    if not args.special_tokens_path:
        args.special_tokens_path = os.path.join(args.base_dataset_path, "special_tokens.txt")

    #* load task_config and special_tokens
    log.info(f"Loading task config and special tokens")
    task_config = json.load(open(args.task_config_path, "r"))
    special_tokens = datatuner_dataset.read_special_tokens(task_config, args.special_tokens_path)
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    #* load dataset as DataLoaders
    log.info(f"Loading dataset from {args.base_dataset_path}")
    train_loader, val_loader = datatuner_dataset.get_data_loaders(
        args.base_dataset_path, task_config, tokenizer, 
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size)
    log.info(f"Train size: {len(train_loader)} - Val size: {len(val_loader)}")

    #* set up optimizer and scheduler
    num_train = len(train_loader.dataset)
    total_train = num_train * args.epochs
    total_steps = ( (num_train // args.train_batch_size) * args.epochs)     # num times that optim.step() will be called
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    log.info(f'Device: {args.device}\n'
             f'Total steps: {total_steps}\n'
             f'Total train (# training examples * epochs): {total_train}\n')

    config_str = "\n"
    for k, v in vars(args).items():
        config_str += f"{k}: {v}\n"
    config_str += f"Save directory: {args.save_dir_path}\n"
    log.info(config_str)

    log.info("Starting training")
    epoch = 0
    step = 0 # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    last_avg_bleu = 0
    best_predictions = []
    best_model_state_dict = None
    while epoch < args.epochs:
        epoch += 1
        #* train
        model.train()
        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                #* forward
                batch_size = len(batch["source_input_ids"])
                loss = forward(model, batch, args.device, pad_token_id=tokenizer.pad_token_id)
                loss_val = loss.item()      # get the item since loss is a tensor
                #* backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad() #! is this needed/harmful? TODO check
                scheduler.step()
                #* log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)
                # tbx.add_scalar('train/loss', loss_val, step)
                # tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

        #* evaluate
        log.info(f'Evaluating at step {step}...')
        intermediate_predictions = []
        bleus = []
        num_val = len(val_loader.dataset)
        model.eval()
        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(val_loader):
                batch_size = len(batch["source_input_ids"])

                # #* evaluation for loss fcn
                # loss, _ = forward(model, batch, args.device)     # loss, logits, but don't need logits
                # loss_meter.update(loss.item(), batch_size)  # loss.item() since it's a tensor

                #* generate for token matches
                source_ids = batch["source_input_ids"].to(args.device, dtype=torch.long)
                source_mask = batch["source_attention_mask"].to(args.device, dtype=torch.long)
                generated_ids = model.generate(
                    source_ids, 
                    attention_mask=source_mask, 
                    max_length=200,
                    num_beams=5,
                    early_stopping=True,
                    num_return_sequences=5,
                ) #! hardcoded length TODO
                #* save for qualitative analysis
                original_data_inputs = tokenizer.batch_decode(batch["source_input_ids"], skip_special_tokens=True)
                original_text_targets = tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
                outputs_decoded = np.array(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
                # they are not divided into batch so we reorder them from (batch size * beam size, sequence size) to (batch, beam, sequence)
                outputs_decoded = outputs_decoded.reshape(-1, 5)
                best_decoded_outputs = []
                for source, generated_beam in zip(original_text_targets, outputs_decoded):
                    highest_bleu = 0
                    highest_sentence_index = 0
                    for index, generated in enumerate(generated_beam):
                        current_bleu = sentence_bleu([source], generated)
                        if current_bleu > highest_bleu:
                            highest_sentence_index = index
                            highest_bleu = current_bleu
                    bleus.append(highest_bleu)
                    best_decoded_outputs.append(generated_beam[highest_sentence_index])
                current_predictions = list(zip(original_data_inputs, original_text_targets, best_decoded_outputs))
                intermediate_predictions.extend(current_predictions)

                #* print one batch of generations for qualitative assessment
                if batch_num == 0:
                    for data, orig_input, actual_output in current_predictions[:1]:
                        log.info(f"\nData: {data}\n"
                                f"\nSource: {orig_input}\n"
                                f"\tGenerated: {actual_output}")

                #* log info
                progress_bar.update(batch_size)
        #* compute the average BLEU score
        current_avg_bleu = sum(bleus)/float(len(bleus))
        log.info(f"\nAverage BLEU at end of epoch {epoch}: {current_avg_bleu:.3f}")
        #* check if the model got worse and stop training in that case
        if current_avg_bleu < last_avg_bleu:
            log.info(f"Stopping training (prev bleu {last_avg_bleu} > curr bleu {current_avg_bleu})")
            break
        #* save the new avg BLUE, predictions and model, since this model is necessarily better
        log.info("Current version of the model is better than the previous ones, saving...")
        last_avg_bleu = current_avg_bleu
        best_predictions = intermediate_predictions
        # state_dict is deepcopied since otherwise it would get updated with the model training
        best_model_state_dict = deepcopy(model.state_dict())
    

    #* create a directory in the save_dir folder called <dataset_name>_<timestamp>
    dataset_name = args.base_dataset_path.split(os.sep)[-1]
    # model_dir_name = f"{dataset_name}_{int(datetime.datetime.now().timestamp())}"
    # full_dir_path = os.path.join(args.save_dir_path, f"{model_dir_name}")
    log.info(f"Training complete, saving predictions and model at {args.save_dir_path}")
    os.makedirs(args.save_dir_path, exist_ok=True)
    #* save predictions
    to_write = ""
    for prediction_pair in best_predictions:
        data, source, generated = prediction_pair
        to_write += f"DATA: {data}\nOG: {source}\nGEN: {generated}\n\n"
    with open(os.path.join(args.save_dir_path, "predictions.txt"), "w") as f:
        f.write(to_write)
    #* save training/model stats
    with open(os.path.join(args.save_dir_path, "stats.txt"), "w") as f:
        f.write(f"Training ended at epoch {epoch}\nLoss {loss_val:.5f}\nAvg final BLEU: {last_avg_bleu:.5f}")
    #* save model
    torch.save({
        "epoch": epoch,
        "model_state_dict": best_model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(args.save_dir_path, "model_params.tar"))
    log.info("Training complete!")


if __name__ == '__main__':
    main()
