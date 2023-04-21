import os
import time

pod_type = os.getenv("GRAPHCORE_POD_TYPE", "pod16")
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/exe_cache/") + "/stablediffusion_to-image"

#from huggingface_hub import notebook_login

#notebook_login()

import torch

from ipu_models import IPUStableDiffusionPipeline

pipe = IPUStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
    ipu_config={
        "executable_cache_dir": executable_cache_dir,
    }
)
print (pipe)
pipe.enable_attention_slicing()

samples = torch.rand([2,4,64,64], dtype=torch.half)
timestep = torch.rand([2], dtype=torch.half)
encoder_hidden_states = torch.rand([2,77,768], dtype=torch.half)

out = pipe.unet(samples, timestep, encoder_hidden_states)

print (out)
