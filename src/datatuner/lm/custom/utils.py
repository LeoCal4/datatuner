import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Line2D


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_grad_flow(named_parameters, save_path: str, max_num_of_params: int = None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if max_num_of_params and len(layers) >= max_num_of_params:
                break
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(f"{save_path}.png")


def plot_grad_flow2(named_parameters, save_path: str, max_num_of_params: int = None):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            if max_num_of_params and len(layers) >= max_num_of_params:
                break
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f"{save_path}.png")


def import_ser_calculator():
    #! this is terrible but there is no other easy way
    import os
    import sys
    custom_dir_path = os.path.dirname(os.path.abspath(__file__))
    ser_dir_path = os.path.join(custom_dir_path, "libs", "data2text-nlp")
    sys.path.append(ser_dir_path)
    import ser_calculator
    return ser_calculator


def format_metrics_compendium(metrics_compendium: Dict[str, float]) -> str:
    ser = f"SER {(metrics_compendium['SER']*100):.3f}% ({metrics_compendium['wrong_slots']})"
    uer = f"UER {(metrics_compendium['UER']*100):.3f}% ({metrics_compendium['wrong_sentences']})"
    other_metrics = "\n".join([
        f"{metric_name.capitalize()}: {metric_value:.3f}"
        for metric_name, metric_value in metrics_compendium.items() 
        if not any(metric_name.starts_with(name) for name in ["SER", "UER", "wrong_"])
        ])
    return f"{ser}\n{uer}\n{other_metrics}"
