import argparse
import json
import logging
import os
from copy import deepcopy
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datatuner.lm.custom import datatuner_dataset, metrics
from datatuner.lm.custom.custom_models import DatatunerModel, GenForwardT5Model
from datatuner.lm.custom import utils
from tqdm import tqdm
from transformers import (GPT2Tokenizer, OPTForCausalLM,
                          T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup, 
                          AutoModelForSeq2SeqLM, AutoTokenizer)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    #* ==== General ====
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    #* ==== Dataset and model paths ====
    parser.add_argument("--base_dataset_path", type=str, help="Path of the dataset.")
    parser.add_argument("--consistency_dataset_path", type=str, default=None, help="Path of the consistency dataset.")
    parser.add_argument("--task_config_path", type=str, help="Path to the tokenization config file")
    parser.add_argument("--special_tokens_path", type=str, default=None, help="Path to the special tokens file")
    parser.add_argument("--train_params_path", type=str, help="JSON file with training parameters.")
    parser.add_argument("--save_dir_path", type=str, default="./save", help="Path to the save directory.")
    #* ==== Model info and params ====
    parser.add_argument("--model_name", type=str, default="t5-base", help="Short name of the model")
    parser.add_argument("--model_type", type=str, default="enc_dec", help="Model type. Either 'enc_dec' or 'dec_only'.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--ser_early_stopping", action="store_true", help="Early stopping based on SER instead of BLEU.")
    parser.add_argument("--text_prefix", type=str, default="from English to Data:", help="The text prefix to prepend to every input sentence.")
    #* ==== SF loss ====
    parser.add_argument("--use_sf_loss", action="store_true", help="Whether to use the semantic fidelity loss or not.")
    parser.add_argument("--sf_loss_alpha", type=float, default=0.0, help="The weight for the semantic fidelity loss.")
    #* ==== DCS loss ====
    parser.add_argument("--use_dcs_loss", action="store_true", help="Whether to use the DCS agumented loss or not.")
    parser.add_argument("--dcs_beta", type=float, default=0.0, help="The weight for the DCS loss.")
    parser.add_argument("--use_custom_forward", action="store_true", help="Use generation-like forward.")
    return parser.parse_args()


def main():
    #* set seed and args
    log.info("Parsing arguments")
    args = parse_arguments()
    utils.set_seed(args.seed)

    #* check if there are any additional training args and add them to args
    log.info(f"Checking additional parameters at {args.train_params_path}")
    if args.train_params_path:
        train_params = json.load(open(args.train_params_path, "r"))
        for param in train_params:
            if not hasattr(args, param):
                setattr(args, param, train_params[param])

    #* load model and tokenizer
    log.info(f"Loading model and tokenizer")
    if "it5" in args.model_name:
        model_name = f"gsarti/{args.model_name}"
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    elif "t5" in args.model_name:
        if not args.use_custom_forward:
            base_model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        else:
            log.info("\tUsing custom generate-like forward")
            base_model = GenForwardT5Model.from_pretrained(args.model_name)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
    # elif "opt" in args.model_name:
        # base_model = OPTForCausalLM.from_pretrained(args.model_name)
        # tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    #* check and eventually update special_tokens_path
    if not args.special_tokens_path:
        args.special_tokens_path = os.path.join(args.base_dataset_path, "special_tokens.txt")

    #* load task_config and special_tokens
    log.info(f"Loading task config and special tokens")
    task_config = json.load(open(args.task_config_path, "r"))
    special_tokens = datatuner_dataset.read_special_tokens(task_config, args.special_tokens_path)
    tokenizer.add_tokens(special_tokens)
    base_model.resize_token_embeddings(len(tokenizer))

    #* get model type
    if not args.model_type:
        if "t5" in args.model_name:
            model_type = "enc_dec"
        # elif "opt" in args.model_name:
        #     model_type = "dec_only"
    else:
        model_type = args.model_type

    #* load dataset as DataLoaders
    log.info(f"Loading dataset from {args.base_dataset_path}")
    train_loader, val_loader = datatuner_dataset.get_data_loaders(
        args.base_dataset_path, task_config, tokenizer,  model_type=model_type,
        text_prefix=args.text_prefix,
        batch_sizes={"train": args.train_batch_size, "validation": args.val_batch_size},
        consistency_dataset_path=args.consistency_dataset_path)
    log.info(f"Train size: {len(train_loader)} - Val size: {len(val_loader)}")

    #* set up optimizer and scheduler
    num_train = len(train_loader.dataset)
    total_train = num_train * args.epochs
    total_steps = ((num_train // args.train_batch_size) * args.epochs) # num times that optim.step() will be called
    # optimizer = Adafactor(
    #     model.parameters(),
    #     lr=1e-3,
    #     eps=(1e-30, 1e-3),
    #     clip_threshold=1.0,
    #     decay_rate=-0.8,
    #     beta1=None,
    #     weight_decay=0.0,
    #     relative_step=False,
    #     scale_parameter=False,
    #     warmup_init=False,
    # )
    dataset_name = args.base_dataset_path.split(os.sep)[-1] #TODO find a cleaner way to do this
    model = DatatunerModel(
        base_model, 
        tokenizer, 
        device=args.device, 
        use_sf_loss=args.use_sf_loss, 
        sf_loss_alpha=args.sf_loss_alpha,
        use_dcs_loss=args.use_dcs_loss,
        dcs_beta=args.dcs_beta,
        current_dataset=dataset_name, #needed for dcs
    )

    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    log.info(f'Device: {args.device}\n'
             f'Total steps: {total_steps}\n'
             f'Total train (# training examples * epochs): {total_train}\n')

    #* log args
    config_str = "\n"
    for k, v in vars(args).items():
        config_str += f"{k}: {v}\n"
    log.info(config_str)

    #* train
    log.info("Starting training")
    #* log sf loss usage
    if args.use_sf_loss:
        log.info(f"\tUsing semantic fidelity loss with alpha={args.sf_loss_alpha}")
    if args.use_dcs_loss:
        log.info(f"\tUsing DCS loss with beta={args.dcs_beta}")
    if args.ser_early_stopping:
        log.info(f"\tEarly stopping using SER")
    epoch = 0
    step = 0 # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
    last_avg_bleu = 0
    last_ser = 100
    best_predictions = []
    best_model_state_dict = None
    while epoch < args.epochs:
        epoch += 1
        log.info(f">>>> Starting epoch {epoch}")
        model.train()
        with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
            for batch_num, batch in enumerate(train_loader):
                #* forward
                loss, logits = model(batch)
                #* backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                #* log info
                batch_size = len(batch["source_input_ids"])
                step += batch_size
                progress_bar.update(batch_size)
                loss_val = loss.item() # get the item since loss is a tensor
                progress_bar.set_postfix(epoch=epoch, loss=loss_val)
                # if batch_num == 0 or batch_num % (batch_size*100) == 0:
                #     softmaxes = F.softmax(logits, dim=-1)
                #     predictions = torch.argmax(softmaxes, -1)
                #     new_pred = []
                #     for pred in predictions.tolist():
                #         if not pred:
                #             continue 
                #         try:
                #             new_pred.append(pred[:pred.index(tokenizer.eos_token_id)])
                #         except ValueError:
                #             new_pred.append(pred)
                #     sentences = tokenizer.batch_decode(new_pred, skip_special_tokens=False)
                #     del predictions
                #     source_data = '\n'.join(batch['source_data_values'][0:5])
                #     log.info(f"\nSource:\n{source_data}")
                #     generated = '\n'.join(sentences[0:5])
                #     log.info(f"\nGenerated:\n{generated}")
                del logits

        #* evaluate
        log.info(f'Evaluating at step {step}...')
        intermediate_predictions = []
        num_val = len(val_loader.dataset)
        original_data_inputs = [] # collecting at each iteration even if it is not needed
        model.eval()
        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(val_loader):
                #* generate for token matches
                generated_ids = model.inference(batch)
                #* save for qualitative analysis
                data_inputs = tokenizer.batch_decode(batch["source_input_ids"], skip_special_tokens=True)
                text_targets = tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
                outputs_decoded = np.array(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
                # they are not divided into batch so we reorder them from (batch size * beam size, sequence size) to (batch, beam, sequence)
                output_beams_decoded = outputs_decoded.reshape(-1, 5)
                # outputs_decoded_no_special_tokens = outputs_decoded_no_special_tokens.reshape(-1, 5)
                # if batch_num % 5 == 0:
                #     log.info(f"gen dec: {outputs_decoded.shape}\n{outputs_decoded[0][0]}")
                original_data_inputs.extend(batch["original_data"])
                current_predictions = list(zip(
                    data_inputs, text_targets, output_beams_decoded
                    ))
                intermediate_predictions.extend(current_predictions)

                #* print one batch of generations for qualitative assessment
                if batch_num == 0:
                    data, orig_input, actual_output = current_predictions[0]
                    log.info(f"\nData: {data}\n"
                            f"\nSource: {orig_input}\n"
                            f"\nGenerated: {actual_output[0]}")
                #* log info
                batch_size = len(batch["source_input_ids"])
                progress_bar.update(batch_size)

        #* compute the average BLEU score
        current_corpus_bleu_score = metrics.corpus_level_bleu(intermediate_predictions)
        log.info(f"BLEU at end of epoch {epoch}: {current_corpus_bleu_score:.3f}")
        #* compute SER
        current_ser_values = metrics.corpus_level_ser(original_data_inputs, intermediate_predictions, dataset_name)
        current_ser = current_ser_values[0]
        log.info(f"SER at end of epoch {epoch}: {(current_ser*100):.3f}%")
        #* check if the model got worse and stop training in that case
        if args.ser_early_stopping:
            if last_ser < current_ser and current_corpus_bleu_score < last_avg_bleu:
                log.info(f"Stopping training (prev SER {last_ser*100}% < curr SER {current_ser*100}%)")
                break
        else:
            if current_corpus_bleu_score < last_avg_bleu:
                log.info(f"Stopping training (prev bleu {last_avg_bleu} > curr bleu {current_corpus_bleu_score})")
                break
        #* save the new avg BLUE, predictions and model, since this model is necessarily better
        log.info("Current version of the model is better than the previous ones, saving...")
        log.info("===============================================================")
        last_avg_bleu = current_corpus_bleu_score
        last_ser_values = current_ser_values
        last_ser = current_ser
        best_loss = loss.item() # saving just the item() to save memory
        best_epoch = epoch
        best_predictions = intermediate_predictions
        # state_dict is deepcopied since otherwise it would get updated with the model training
        best_model_state_dict = deepcopy(model.state_dict())

    #* create the save/output folder
    log.info(f"Saving predictions and model at {args.save_dir_path}")
    os.makedirs(args.save_dir_path, exist_ok=True)
    #* save predictions
    predictions = []
    for prediction_tuple in best_predictions:
        data, source, generated_beam = prediction_tuple
        predictions.append({"data": data, "ref": source, "gen": generated_beam.tolist()})
    with open(os.path.join(args.save_dir_path, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, sort_keys=False, indent=4, ensure_ascii=False)
    #* calculate metrics
    metrics_compendium = metrics.create_metrics_compendium(
        best_predictions,
        precomputed_ser=last_ser_values,
        precomputed_bleu=last_avg_bleu
    )
    #* save training/model stats
    metrics_compendium["loss"] = best_loss
    metrics_compendium["epoch"] = best_epoch
    with open(os.path.join(args.save_dir_path, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_compendium, f, sort_keys=False, indent=4, ensure_ascii=False)
    #* save model
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss #! REMOVED to save memory
    }, os.path.join(args.save_dir_path, "model_params.tar"))
    #* print results
    formatted_metrics_compendium = utils.format_metrics_compendium(metrics_compendium)
    formatted_metrics_compendium = f"Training completed!\n{formatted_metrics_compendium}"
    log.info(formatted_metrics_compendium)


if __name__ == '__main__':
    main()
