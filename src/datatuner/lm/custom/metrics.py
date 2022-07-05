from typing import List, Tuple

import sacrebleu 

def corpus_level_bleu(model_inputs_and_outputs: List[Tuple[str]]):
    """Aggregate all the sentences which share the same original data, then calculate the corpus-level bleu score.

    This is a slightly simplified version of bleu(...) found in lm/metrics.py

    Args:
        inputs_and_outputs (List[List[str]]): list of a zip containing both the model's inputs and outputs,
            specifically: input data, original sentence, default choice generated sentence 

    Returns:
        Dict[str, float]: dict containing the corpus-level bleu and the number of sentences taken into account 
    """
    grouped_items = {}
    max_num_of_og_sentences = 1
    for data, original_sentence, generated_sentence in model_inputs_and_outputs:
        #* take the first sentence among the generated ones, if there are more than one
        if type(generated_sentence) == list or type(generated_sentence) == tuple:
            generated_sentence = generated_sentence[0]

        #* lowercase sentences
        generated_sentence = generated_sentence.lower()
        original_sentence = original_sentence.lower()

        #* group the og and gen sentences by the values of their other keys (basically their data)
        if data in grouped_items:
            grouped_items[data]["original"].append(original_sentence)
            #? this ends up considering just the last found sentence generated by certain data 
            #?  this probably made sense in datatuner since the outputs were always the same given the same input data 
            grouped_items[data]["prediction"] = generated_sentence
            if len(grouped_items[data]["original"]) > max_num_of_og_sentences:
                max_num_of_og_sentences = len(grouped_items[data]["original"])
        else:
            grouped_items[data] = {"original": [original_sentence], "prediction": generated_sentence}

    all_predictions = []
    all_originals = [[] for _ in range(max_num_of_og_sentences)]
    for item in grouped_items.values():
        all_predictions.append(item["prediction"])
        for i in range(max_num_of_og_sentences):
            try:
                all_originals[i].append(item["original"][i])
            except:
                all_originals[i].append("")

    return sacrebleu.corpus_bleu(all_predictions, all_originals).score
