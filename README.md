# <Paper title>

This repo is code used for experiments in the paper [2Mamba2Furious: Linear in complexity, competitive in accuracy](https://arxiv.org/abs/2602.17363).

Unfiltered experiments conducted before writing the paper are supplied at [https://wandb.ai/gmongaras1/Gated_Attention_V2](https://wandb.ai/gmongaras1/Gated_Attention_V2) [https://wandb.ai/gmongaras1/Mamba_buildup](https://wandb.ai/gmongaras1/Mamba_buildup). Experiments from the paper as well as other experiments that never got into the paper are supplied at [https://wandb.ai/gmongaras1/Mamba_Squared_Experiemnts](https://wandb.ai/gmongaras1/Mamba_Squared_Experiemnts)



# Setup

This repo was trained with python 3.10. Other versions may or may not work.

To setup, first ensure you have cuda properly setup. This can be checked by running `nvidia-smi` and `nvcc -V`.

Create a virtual environment with 
```
python -m venv TwoMambaEnv
source TwoMambaEnv/bin/activate
```

Install the requirements
```
pip install -r requirements.txt
```

Then, install the version of torch for your system at `https://pytorch.org/get-started/locally/`. This repo was run on torch `2.6.0` with cuda `11.8`. Your system will likely need to use a different version of cuda. The following command install torch `2.6.0` for cuda `11.8`:
```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

I use causal conv 1d a lot. You can donload that at [https://github.com/Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). I used version 1.5.4 as I think that was the needed version for my cuda. You can find the version for `1.6.0` (latest as of now) here [https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.0](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.0). Manual download is likely required.

If you plan to use the traditional mamba model, you will have to download that from the mamba_ssm repo here [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba). Download is similar to causal conv 1d, you will likely have to manually download the wheel. I used version `2.2.6.post3`.





# Running the script

The script can be run with the following:

```
torchrun --nproc_per_node=2 --master-port $PORT GPT_Trainer/train.py $SEQLEN $WANDB_NAME $RUN_NAME
```

where `$PORT` is just an arbitrary open port number, `$SEQLEN` is the sequence length to train on, `$WANDB_NAME` is the name of the wandb run, and `$RUN_NAME` is the local run name.




# Training and Inference

## Training

Training can be done by running `GPT_Trainer/train.py` script. An example of how to run this script is in `runjob.sh`. A few of the params are command-line specific as it helped me launch a bunch of runs with various configurations on the cluster I was using. The following are those params:
1. `seq_len` - Sequnce length to train on. I used 2048, 4096, and 8192.
2. `run_name` - Wandb name for this run. Model checkpoints (along with optimizer states for training from a checkpoint) are saved to `models/{run_name}`. Note that checkpoints override each other as I didn't have much memory to work with.
3. `attention_type` - Attention type to run. The types you can choose from can be found in `GPT_Trainer/LlamaDecoderLayerClean.py` in the `configs` dictionary.

The following are params controlled in the train script:
- `batch_size` - Global batch size across all GPUs. A batch size of 32 across 2 GPUs would mean a batch size of 16 on each GPU.
- `learning_rate` - Model learning rate
- `warmup_steps` - Number of steps to linearly increase the learning rate from 0 to `learning_rate`. After the number of steps reaches `warmup_steps`, the learning rate is linearly decreased to 0 and will hit 0 once `num_steps` update steps has been reached.
- `num_steps` - Total number of steps the model will be trained for.
- `num_steps_early_stop` - Stops the model early at this many steps. The learning rate scheduler is not changed.
- `dev` - Keep this at `gpu`
- `wandb_name` - Wandb will log the run under this name. Note that the project is defaulted to `run_name` from the cmd line params.
- `log_steps` - Wandb will log the model loss every `log_steps` number of steps.
- `use_amp` - True to use AMP with bfloat16, False to stay in float32.
- `attention_type` - Type of attention this model will use. This value defaults to `attention_type` from the cmd line params.
- `dataset` - Name of the dataset to load in. Can be one of `gmongaras/EleutherAI_the_pile_deduplicated`, `gmongaras/SlimPajama-627B_Reupload`, `HuggingFaceFW/fineweb`.
- `mlp_type` - THe MLP type to use. We stick with `normal` which uses the normal, gated MLP block in llama. `gelu` swaps this with a GELU MLP without a gate.
- `clipping_value` - `None` for no clipping, a float value to perform gradient clipping with said value.
- `weight_decay` - Normal optimizer weight decay param
- `model_save_path` - Local path to save model checkpoints to. Defaults to `models/{run_name}`.
- `num_save_steps` - Every `num_save_steps`, the current model will be saved along with the states of the optimizer and scheduler.
- `keep_dataset_in_mem` - Keep this `False`
- `model_max_length` - The max sequence length to train the model with. Defualts to `seq_len` from the cmd line params.
- `test_per` - Percentage of data to make test data.
- `num_steps_test` - Every `num_steps_test`, the model will stop trained, will iterate over all test data, and calculate metrics logged to wandb.
- `model_size` - `small` for the small model (~300 million params), `medium` for the large model (~700 million params).
- `test_loss` - `True` to test the model on test data, `False` to skip this and just train the model.

The base model is llama 2. We just swap out the blocks with custom blocks. The code for these blocks can be found in the `GPT_Trainer/LlamaDecoderLayerClean.py` script. Additionally, this script has all the `attention_type` options.

The training actually happens in `Trainer.py`.

To start training from a pretrained checkpoint, download models as noted below, set `load_checkpoint` to True, and `checkpoint_path` to the directory containing the `.pt` files. Note that you will need to download all files unlike inference which just requires the model.


## Inference

To test models, `infer2.py` can perform inference using pretrained models. First, make sure you have downloaded the pretrained model you want to use, as noted below. This script is not really optimized, though it does have "efficient" modes for both softmax and `2mamba` by setting `use_efficient` to True.

### Downloading models for inference and training

We supply pretrained models at [https://huggingface.co/collections/gmongaras/2mamba2furious](https://huggingface.co/collections/gmongaras/2mamba2furious-linear-in-complexity). These models are long-running 400K step models trained for NIAH. Download the models directly and throw them in some directory. For example, you could download `medium_8192sl_gpu_64bs__squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__att2` and put it in the `models/medium_8192sl_gpu_64bs__squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__att2` directory. As noted above, these files can be downloaded and used to continue pretraining, though you will have to download all checkpoints. For inference, you do not need `optimizer.pt` or `scheduler.pt`. 

Note that efficient inference for 2Mamba requires a custom kernel created at [https://github.com/gmongaras/Triton-Efficient-Kronecker-Product](https://github.com/gmongaras/Triton-Efficient-Kronecker-Product).


# Experiment Info

Unless otherwise mentioned, the below are the parameters we used in our models. As our base model is llama 2, RoPE is used on the attention matrix and the MLPs follow SwiGLU.

- batch size - 32
- learning rate - 1e-4
- warmup steps - 10,000
- warmup type - linear warmup from 0, linear decay
- num steps - 1,000,000
- num steps early stop - 100,000
- AMP - enabled
- Weight decay - 0.01
- Max sequence length - 2048, 4096, and 9182
- Test percentage - 0.001
- Optimizer - AdamW
- Adam betas - 0.9 and 0.999
- Hidden size - 1024 (1536 for the medium model)
- MLP intermediate size - 1024 (3072 for the medium model)
- Num attention heads - 16 (24 for the medium model)
- Num hidden layers - 20 (27 for the medium model)
- Tokenizer - llama2-7b-hf
- Gradient clipping - No clipping for all experiments

Most models was trained for a maximum of 2 days. The long NIAH experiemnts required about a week of training. Most experiments, we use distributed data parallel to train on two 80 GB, A100 GPUs with the exception of the medium model at 8192 seq len trained on 16 GPUs.

All experiemnts in the paper are controlled through the config in `GPT_Trainer/LlamaDecoderLayerClean.py`, however many other experiments were ran that show up in the other `LlamaDecoderLayer...` files if one is interested. Some mamba tests can be found in `mamba_test` and `mamba_test_`.


## LlamaDecoderLayerClean

This python file contains a lot of the experiments we performed. The other LlamaDecoderLayer contains messier experiemnts we ran. The most notable 4 `attention_type` configurations are:
1. `squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2` - The `2Mamba` model our analysis ended up with. Note that the method with value discretization is at `squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2__dt_on_values`, however note that we neded up removing value discretization as noted in the paper.
2. `exp__sm_norm__A_mask_type_neg_softplus__in_conv_k_2` - The `2Mamba-E` model with the exponential.
3. `softmax` - Good ol softmax attn
4. `mamba` - Mamba "attention" from the official mamba repo `https://github.com/state-spaces/mamba`


## NIAH

NIAH can be run with `GPT_Trainer/niah.py`. To reproduce our results, use one of the provided pretrained checkpoints. The dataset can be found at [nanotron/simple_needle_in_a_hay_stack](https://huggingface.co/datasets/nanotron/simple_needle_in_a_hay_stack). Note that since the cluster I ran NIAH on would kill my experiments, I only ran up to 1000 runs. The total number of rows is 12K. The dataset had a 16K character test, however these were removed for memory requirements. The random subset was seeded so our results should be reproducable.

The NIAH heatmap can be generated with `niah_and_hs/niah_heatmap.py`. We hardcode the results printed from the `niah.py` script.

... As you can probably tell this was kinda done at the very end of the the project. I think it would be interesting to dive into more. I'm transparent about the results we have so while I do think it's cool that these results show simialr context usage to softmax, NIAH and context usage should be looked into more before any conclusions are made about context usage.

## Paper figures

All included figures can be found and reproduced using the script found in `wandb_graphs/`. We create a tool at [https://github.com/gmongaras/Wandb_Plotting_Tool](https://github.com/gmongaras/Wandb_Plotting_Tool) to easily create graphs. Annoyingly arxiv required pngs, not svgs. We don't include these but they can easily be created by running `https://github.com/gmongaras/2Mamba2Furious/blob/main/wandb_graphs/my_main.py`

## Kernels

A bunch of kernels to try to get things working are found in `kernel/`. All of them are quite bad. They were mostly creted just as a proof of concept as, memory wise, full quadratic attention is impossible to run, even with gradient checkpointing. The ones I ended up using are imported and used in `GPT_Trainer/LlamaDecoderLayerClean.py`. I wouldn't use these terrible kernels outside of experimenting with this repo lol.


# Datasets

The fineweb dataset was used for most expierments. We specifically use the "CC-MAIN-2024-51" version of this dataset. The Pile and SlimPajama were also used and have been reuploaded to utilize faster loading speeds of the more recent huggingfae library.
- [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/CC-MAIN-2024-51)
- [The Pile (Reuploaded)](https://huggingface.co/datasets/gmongaras/EleutherAI_the_pile_deduplicated)
- [The Pile (Original)](https://huggingface.co/datasets/EleutherAI/pile)
- [SlimPajama (Reuplaoded)](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload)
- [SlimPajama (Original)](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
