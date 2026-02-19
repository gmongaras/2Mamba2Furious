import torch
from torch import nn
import transformers
import datasets
import os
from tqdm import tqdm
import safetensors
from Triton_Efficient_Kronecker_Product.kron import kron
from GPT_Trainer.Trainer import get_model
from datasets import load_dataset
import numpy as np

try:
    from GPT_Trainer.multi_gpu_helpers import is_main_process
    from GPT_Trainer.LlamaDecoderLayerClean import LlamaDecoderLayer
    from GPT_Trainer.infer2 import infer_
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from LlamaDecoderLayerClean import LlamaDecoderLayer
    from infer2 import infer_



@torch.no_grad()
def infer():
    # Path to the model
    model_path = "models/medium_8192sl_gpu_64bs__squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__att2_"
    # model_path = "models/medium_8192sl_gpu_64bs__softmax_"
    # model_path = "models/medium_8192sl_gpu_64bs__mamba_"
    device = "cuda:0"
    use_efficient = False
    sample = False
    no_eos_token = False
    num_evals = 1000


    # Read token from .env file
    with open(".env", "r") as f:
        token = f.read().strip()

    # Tokenizer
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, cache_dir="GPT_Trainer/llama2", token=token)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", use_fast=False, cache_dir="GPT_Trainer/gpt-j-6B")
    except OSError:
        raise FileNotFoundError("Token not found in .env file or user does not have access to Llama 2 weights with that token. Please add your Hugging Face token to the .env file.")
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token = torch.tensor([tokenizer.pad_token_id])

    # Load the config
    config = torch.load(os.path.join(model_path, "config.pt"))

    # Set model
    model = get_model(config["model_size"], config["model_max_length"], tokenizer.vocab_size, config["attention_type"])
    # model = transformers.LlamaForCausalLM(config=transformers.LlamaConfig.from_dict(config)).cuda()

    
    # Replace all self attention layers with the cosine attention layer
    for i, layer in enumerate(model.model.layers):
        old = layer
        model.model.layers[i] = LlamaDecoderLayer(model.config, layer_idx=i).to(layer.self_attn.q_proj.weight.device)
        model.model.layers[i].self_attn.layer_num = i
        model.model.layers[i].self_attn.use_efficient = use_efficient
        model.model.layers[i].self_attn.is_inference = True
        del old

    # Load in params
    model.load_state_dict(safetensors.torch.load_file(model_path + "/model.safetensors"), strict=True)
    model.eval()
    
    
    # Clear cache
    torch.cuda.empty_cache()
        
    model = model.cuda()
    model.eval()
    
    # Get dataset
    cache_path = "cache"
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_CACHE"] = cache_path
    niah_dataset = load_dataset(
        "nanotron/simple_needle_in_a_hay_stack", 
        cache_dir=cache_path, 
        download_config=datasets.DownloadConfig(cache_dir=cache_path),
        split="train",
    )
    
    np.random.seed(420)
    idxs = np.arange(0, niah_dataset.num_rows)
    np.random.shuffle(idxs)
    
    # Skip 32K lengths as memory blows up and I don't feel like dealing with that
    context_lengths = niah_dataset["context_length"]
    idxs = [i for i in idxs if context_lengths[i] < 32768]
    
    # Get a subset
    idxs = idxs[:num_evals]
    
    # Iter over all rows. Save the proportion right for each length.
    acc_tracker = {}
    for i, idx in enumerate(tqdm(idxs)):
        row_ = niah_dataset._getitem(int(idx))
        ctx_len = row_["context_length"]
        
        # Get output
        output = infer_(row_["prompt"], 10, model, tokenizer, sample, use_efficient, no_eos_token, False)
        
        # Parse output and add to totals
        pred = tokenizer.decode(output[-10:]).split(" ")[0]
        pred = "".join([i for i in pred if i.isnumeric()])
        correct = int(pred == str(row_["answer"]))
        try:
            acc_tracker[ctx_len][0] += correct
            acc_tracker[ctx_len][1] += 1
        except KeyError:
            acc_tracker[ctx_len] = [correct, 1]
    
    for ctx_len, accs in acc_tracker.items():
        print(f"ctx_len: {ctx_len} - acc: {accs[0]/accs[1]} - total correct: {accs[0]} - total: {accs[1]}")
    
    
    
if __name__ == "__main__":
    infer()
