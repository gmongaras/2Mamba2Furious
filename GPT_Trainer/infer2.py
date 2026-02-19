import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import safetensors
from Triton_Efficient_Kronecker_Product.kron import kron
from GPT_Trainer.Trainer import get_model

# import random
# import numpy as np
# def set_seed(seed_value=42):
#     """Set seeds for reproducibility."""
#     # 1. Set seed for Python's built-in random module
#     random.seed(seed_value)
#     # Set seed for hash to ensure consistent environment
#     os.environ['PYTHONHASHSEED'] = str(seed_value)
#     # 2. Set seed for NumPy
#     np.random.seed(seed_value)
#     # 3. Set seed for PyTorch (both CPU and GPU)
#     torch.manual_seed(seed_value)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed_value)
#         torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
#     # 4. Configure PyTorch to use deterministic algorithms
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False # Disabling benchmarking helps ensure determinism
#     # Note: torch.use_deterministic_algorithms(True) can also be used in newer versions
# set_seed(1234) # You can use any integer seed value


try:
    from GPT_Trainer.multi_gpu_helpers import is_main_process
    from GPT_Trainer.LlamaDecoderLayerClean import LlamaDecoderLayer
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from LlamaDecoderLayerClean import LlamaDecoderLayer



def infer_(prompt, num_outputs, model, tokenizer, sample, use_efficient, no_eos_token=False, decode_output=True):
    # Reset states
    for layer in model.model.layers:
        layer.self_attn.reset_inference_states()
        
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
        
    output_tokens = list(inputs["input_ids"][0].detach().cpu().numpy())
    for i in range(len(inputs["input_ids"][0]), len(inputs["input_ids"][0])+num_outputs):
        # Get the logits
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        outputs = model(input_ids, attention_mask=attention_mask)
            
        # Get the predicted next word
        logits = outputs.logits[0, -1]
        
        # Set prob of <|endoftext|> to 0
        if no_eos_token:
            logits[50256] = -float("inf")
        
        # Sample or argmax
        if sample:
            dist = torch.distributions.Categorical(logits=logits)
            next_word = dist.sample()
        else:
            next_word = logits.argmax(-1)
        
        if next_word == tokenizer.eos_token_id:
            break
        
        # Add the next word to the input
        output_tokens.append(next_word.item())
        if use_efficient:
            inputs["input_ids"] = next_word.unsqueeze(0).unsqueeze(0)
            inputs["attention_mask"] = torch.ones(1, 1).cuda()
        else:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_word.unsqueeze(0).unsqueeze(0)], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(1, 1).cuda()], dim=1)
        if i % 10 == 0:
            # print(tokenizer.decode(output_tokens))
            pass
    
    if decode_output:
        return tokenizer.decode(output_tokens)
    return output_tokens



@torch.no_grad()
def infer():
    # Path to the model
    model_path = "models/medium_8192sl_gpu_64bs__squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2_"
    # model_path = "models/medium_8192sl_gpu_64bs__softmax_"
    device = "cuda:0"
    model_max_length = 8192
    use_efficient = True
    sample = True
    no_eos_token = False
    decode_output = True


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
    
    # Set max sequence length
    tokenizer.model_max_length = model_max_length

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
    
    # Number of parameters in billions
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000_000
    print(f"Number of parameters: {num_params:.2f}B")
        
    model = model.cuda()
    model.eval()
    
    # Load the tokenizer
    # tokenizer = torch.load(os.path.join(model_path, "tokenizer.pt"))  
            
    # inference
    sentence = "Tell me about Ravens."
    # sentence = "The fox"
    # sentence = "The best bat in February is not always the best bat in June MLB Draft. Projection is a fickle thing and scouts are often divided on"
#     sentence = """
# There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. The pass key is 9330. Remember it. 9330 is the pass key. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is 

# What is the pass key? The pass key is """.strip() + " "
#     sentence = """
# There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. The pass key is 9235. Remember it. 9235 is the pass key. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is winding. Into the woods. Out and back.The river is wide. The current is strong. The banks are muddy. Down the stream. Back and forth.The ocean is vast. The waves are calming. The sand is warm. Onward we sail. From coast to coast. Here we go. There and back again.The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.The mountain is high. The snow is cold. The air is thin. Upward we climb. From base to peak.The forest is dense. The trees are tall. The path is windin 

# What is the pass key? The pass key is """.strip() + " "
    
    # labels = inputs["labels"].to(model.device)
    
    decoded = infer_(sentence, model_max_length, model, tokenizer, sample, use_efficient, no_eos_token, decode_output)
        
    # Decode the output
    # decoded = tokenizer.decode(output_tokens)
    
    print(decoded)
    
    
    
if __name__ == "__main__":
    infer()