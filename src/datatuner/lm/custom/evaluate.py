import argparse
import json
import logging
import os
import random
from typing import *

import numpy as np
import torch
from datatuner.lm.custom import datatuner_dataset, metrics
from datatuner.lm.custom.custom_models import CustomT5Model, CustomOPTModel
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, OPTForCausalLM, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint to be loaded")    
    parser.add_argument("--base_dataset_path", type=str, help="Path of the dataset.")
    parser.add_argument("--task_config_path", type=str, help="Path to the tokenization config file")
    parser.add_argument("--special_tokens_path", type=str, default=None, help="Path to the special tokens file")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--save_dir_path", type=str, default="./save", help="Path to the save directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Short name of the model")
    parser.add_argument("--model_type", type=str, default="enc_dec", help="Model type. Either 'enc_dec' or 'dec_only'.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    #* set seed and args
    log.info("Parsing arguments")
    args = parse_arguments()
    set_seed(args.seed)

    #* load base model and tokenizer
    # TODO refactor
    log.info(f"Loading base model and tokenizer")
    if "t5" in args.model_name:
        base_model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, model_max_length=512)
    elif "opt" in args.model_name:
        base_model = OPTForCausalLM.from_pretrained(args.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    #* check and eventually update special_tokens_path
    if not args.special_tokens_path:
        args.special_tokens_path = os.path.join(args.base_dataset_path, "special_tokens.txt")

    #* load task_config and special_tokens
    log.info(f"Loading task config and special tokens")
    task_config = json.load(open(args.task_config_path, "r"))
    special_tokens = datatuner_dataset.read_special_tokens(task_config, args.special_tokens_path)
    tokenizer.add_tokens(special_tokens)

    #* resize model embeddings and load checkpoint
    log.info(f"Loading model checkpoint")
    base_model.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(args.checkpoint_path)
    if "t5" in args.model_name:
        model = CustomT5Model(
            base_model, 
            tokenizer, 
            device=args.device, 
            # use_sf_loss=args.use_sf_loss, 
            # sf_loss_alpha=args.sf_loss_alpha
        )
    elif "opt" in args.model_name:
        model = CustomOPTModel(
            base_model, 
            tokenizer, 
            device=args.device, 
            # use_sf_loss=args.use_sf_loss, 
            # sf_loss_alpha=args.sf_loss_alpha
        )
    else:
        raise ValueError(f"Cannot find custom model for {args.model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    #* get model type
    if not args.model_type:
        if "t5" in args.model_name:
            model_type = "enc_dec"
        elif "opt" in args.model_name:
            model_type = "dec_only"
    else:
        model_type = args.model_type

    #* load dataset as DataLoaders
    log.info(f"Loading dataset from {args.base_dataset_path}")
    test_loader = datatuner_dataset.get_data_loaders(
        args.base_dataset_path, task_config, tokenizer, model_type,
        dataset_types=["test"], batch_sizes={"test": args.test_batch_size})
    log.info(f"Test size: {len(test_loader)}")

    #* log args
    config_str = "\n"
    for k, v in vars(args).items():
        config_str += f"{k}: {v}\n"
    log.info(config_str)

    #* evaluate
    log.info(f"Starting evaluation")
    inputs_and_predictions = []
    best_choice_bleus = []
    default_choice_bleus = []
    num_test = len(test_loader.dataset)
    model.eval()
    with torch.no_grad(), tqdm(total=num_test) as progress_bar:
        for batch_num, batch in enumerate(test_loader):
            batch_size = len(batch["source_input_ids"])
            #* generate for token matches
            generated_ids = model.inference(batch)
            #* save for qualitative analysis
            original_data_inputs = tokenizer.batch_decode(batch["source_input_ids"], skip_special_tokens=True)
            original_text_targets = tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
            outputs_decoded = np.array(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            # they are not divided into batch so we reorder them from (batch size * beam size, sequence size) to (batch, beam, sequence)
            output_beams_decoded = outputs_decoded.reshape(-1, 5)
            current_predictions = list(zip(
                original_data_inputs, original_text_targets, output_beams_decoded
                ))
            inputs_and_predictions.extend(current_predictions)

            #* print one batch of generations for qualitative assessment
            if batch_num == 0:
                data, orig_input, actual_output = current_predictions[0]
                log.info(f"\nData: {data}\n"
                        f"\nSource: {orig_input}\n"
                        f"\nGenerated: {actual_output}")
            #* log info
            progress_bar.update(batch_size)

    #* create the save/output folder (by default it's the one in which the checkpoint is found, so it's already there)
    log.info(f"Saving predictions and results at {args.save_dir_path}")
    os.makedirs(args.save_dir_path, exist_ok=True)
    #* save predictions and organize data for bleu
    predictions = []
    for prediction_tuple in inputs_and_predictions:
        data, source, generated_beam = prediction_tuple
        predictions.append({"data": data, "ref": source, "gen": generated_beam.tolist()})
    with open(os.path.join(args.save_dir_path, "test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, sort_keys=False, indent=4, ensure_ascii=False)
    #* calculate metrics and save them
    corpus_bleu_score = metrics.corpus_level_bleu(inputs_and_predictions)
    chrf_score = metrics.corpus_level_chrf(inputs_and_predictions)
    ter_score = metrics.corpus_level_ter(inputs_and_predictions)
    rouge_score = metrics.corpus_level_rogue(inputs_and_predictions)
    meteor_score = metrics.corpus_level_meteor(inputs_and_predictions)
    metric_compendium = f"""Evaluation completed!
BLEU: {corpus_bleu_score:.5f}
Rouge: {rouge_score:.5f}
Meteor: {meteor_score:.5f}
CHRF: {chrf_score:.5f}
TER: {ter_score:.5f}"""
    with open(os.path.join(args.save_dir_path, "test_stats.txt"), "w") as f:
        f.write(metric_compendium)
    log.info(metric_compendium)


if __name__ == "__main__":
    main()
