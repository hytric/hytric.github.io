import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from safetensors.torch import load_file
from pathlib import Path
import argparse
import os
from collections import Counter, defaultdict
import tempfile
from cgip.cgip import CGIPModel
from tqdm import tqdm
from PIL import Image
import numpy as np

from ip_adapter.ip_adapter import SgProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
    
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
def set_dataset(args, vocab, text):
    # image, objs, boxes, triples
    image = torch.zeros(3, 256, 256)
    objs = torch.tensor([35, 118, 3, 134, 4, 25, 106, 106, 90, 5, 37, 113, 0])
    boxes = torch.zeros(13, 4)
    triples = torch.tensor([
        [2, 3, 1],
        [0, 3, 3],
        [2, 3, 1],
        [2, 12, 4],
        [0, 0, 12],
        [1, 0, 12],
        [2, 0, 12],
        [3, 0, 12],
        [4, 0, 12],
        [5, 0, 12],
        [6, 0, 12],
        [7, 0, 12],
        [8, 0, 12],
        [9, 0, 12],
        [10, 0, 12],
        [11, 0, 12]
    ])
    
    path = os.path.abspath(os.path.join(args.output_dir, f"{text}"))
    save_graph_info(vocab, objs, triples, path)
    draw_scene_graph(objs, triples, vocab, output_filename=args.output_dir)
    
    return vg_collate_fn(image, objs, boxes, triples)

def vg_collate_fn(img, objs, boxes, triples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    img = img[None].to(device)
    objs = objs.to(device)
    boxes = boxes.to(device)
    triples = triples.clone().to(device)

    O, T = objs.size(0), triples.size(0)

    obj_to_img = torch.zeros(O, dtype=torch.long, device=device)
    triple_to_img = torch.zeros(T, dtype=torch.long, device=device) 

    out = (img, objs, boxes, triples, obj_to_img, triple_to_img)
    return out


   
def save_graph_info(vocab, objs, triples, path):
    f = open(path, 'w')
    data = f'''
    objs : 
    {objs}
    
    
    triples :
    {triples}
    
    
    ### vocab ###
    {vocab['object_name_to_idx']}
    
    {vocab["attribute_name_to_idx"]}
    
    {vocab["pred_name_to_idx"]}
    '''
    f.write(data)
    f.close() 

def draw_scene_graph(objs, triples, vocab=None, **kwargs):
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            # p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            p = triples[i, 1].item()
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]

    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        p = vocab['pred_idx_to_name'][p]
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    return None

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Check file extension and load state_dict accordingly
        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=True)


        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

def load_models(args):
    # Load tokenizer and models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # Initialize IPAdapter and other necessary components
    sg_encoder = CGIPModel(num_objs=179, num_preds=46, layers=5, width=512, embed_dim=512, ckpt_path='/home/jskim/T2I/IP-Adapter/pretrained/sip_vg.pt')
    #ip-adapter
    image_proj_model = SgProjModel()
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    # Set models to evaluation mode
    vae.eval()
    unet.eval()
    text_encoder.eval()
    sg_encoder.eval()
    ip_adapter.eval()

    return tokenizer, text_encoder, vae, unet, sg_encoder, ip_adapter

def infer(args, input_text, graph_info):
    # Load models
    tokenizer, text_encoder, vae, unet, sg_encoder, ip_adapter = load_models(args)
    
    # Move models to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    sg_encoder.to(device)
    ip_adapter.to(device)
    
    # Preprocess input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    text_embeddings = text_encoder(inputs.input_ids)[0]

    # Create unconditional (zero) embeddings for CFG
    uncond_inputs = tokenizer([""], return_tensors="pt").to(device)
    uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0]

    # Encode the scene graph
    with torch.no_grad():
        c_globals, c_locals = sg_encoder(graph_info)
    
    # Prepare image embeddings
    image_embeds = (c_globals, c_locals)
    
    # Initialize DDIM scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # Set the number of inference steps
    num_inference_steps = 50
    noise_scheduler.set_timesteps(num_inference_steps)

    # Generate initial latent
    latent_shape = (1, unet.in_channels, 64, 64)  # Assuming a latent shape for 64x64 images, adjust as needed
    latents = torch.randn(latent_shape, device=device)  # Start with random noise

    # Set the number of denoising steps
    timesteps = noise_scheduler.timesteps  # Use the scheduler's predefined timesteps

    # Guidance scale for CFG
    guidance_scale = 7.5  # Adjust this scale factor to control guidance strength

    # Denoising loop with Classifier-Free Guidance
    for timestep in tqdm(timesteps, desc="Denoising Progress"):
        with torch.no_grad():
            # Unconditional prediction
            noisy_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), timestep)
            uncond_pred = ip_adapter(noisy_latents, timestep, uncond_embeddings, image_embeds)
            
            # Conditional prediction
            cond_pred = ip_adapter(noisy_latents, timestep, text_embeddings, image_embeds)
        
        # CFG formula: pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        
        # Update latents using the DDIM step
        latents = noise_scheduler.step(cond_pred, timestep, latents).prev_sample

    # Decode the final latent representation to get the image
    with torch.no_grad():
        generated_image = vae.decode(latents)

    return generated_image

def main():
    parser = argparse.ArgumentParser(description="Inference script for the trained model.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/home/jskim/T2V/AnimateDiff/models/stable-diffusion-v1-5", help="Path to pretrained model directory")
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default="/home/jskim/T2I/IP-Adapter/result/20240912-052023/checkpoint-10.pt", help="Path to pretrained IP Adapter")
    parser.add_argument("--vocab_path", type=str, default="/home/jskim/Graph_VIdeo_generation/dataset/vg/vocab.json", help="Path to pretrained IP Adapter")
    parser.add_argument("--output_dir", type=str, default="/home/jskim/T2I/IP-Adapter/inf_result", help="Path to pretrained IP Adapter")
    
    args = parser.parse_args()
    
    import json
    
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    # Example input (replace with your actual data)
    input_text = "A student are studying in the classroom"
    graph_info = set_dataset(args, vocab, input_text)

    # Run inference
    output_image = infer(args, input_text, graph_info)
    
    # Save image
    output_image_tensor = output_image.sample  
    output_image_tensor = output_image_tensor.squeeze(0).cpu().numpy() 
    output_image_np = (output_image_tensor * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)  
    pil_image = Image.fromarray(output_image_np)

    output_path = os.path.abspath(os.path.join(args.output_dir, "output_image.png"))
    pil_image.save(output_path)
    print(f"Generated image saved to {output_path}")
    
if __name__ == "__main__":
    main()
