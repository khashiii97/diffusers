import torch
import requests
from PIL import Image

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
# Load the pipeline

# torch.Size([2, 25, 8, 72, 128])
# torch.Size([])
# torch.Size([2, 1, 1024])
# torch.Size([2, 3])

# def test_svd_unet(unet):
    
#     latent_model_input = torch.zeros(size = [2, 25, 8, 72, 128], dtype= torch.float16)
#     t = torch.zeros(size = [], dtype= torch.float32)
    
#     image_embeddings = torch.zeros(size = [2, 1, 1024], dtype = torch.float16)
#     added_time_ids = torch.zeros(size = [2, 3], dtype = torch.float16)
#     noise_pred = unet(
#                     latent_model_input,
#                     t,
#                     encoder_hidden_states=image_embeddings,
#                     added_time_ids=added_time_ids,
#                     return_dict=False,
#                 )
#     print(noise_pred.shape)
def merge_unets(pipeline_svd, pipeline_zero123): # replaces the spatial module in unet spatio temporal of svd with modules from unet 2d condition in zero123++
     
     
     #down block


    ####################################
     #    CrossAttnDownBlockSpatioTemporal <- CrossAttnDownBlock2D

    

    #attentions
    pipeline_svd.unet.down_blocks[0].attentions[0].transformer_blocks= pipeline_zero123.unet.down_blocks[0].attentions[0].transformer_blocks
    
    pipeline_svd.unet.down_blocks[0].attentions[1].transformer_blocks = pipeline_zero123.unet.down_blocks[0].attentions[1].transformer_blocks


    # resnets
    pipeline_svd.unet.down_blocks[0].resnets[0].spatial_res_block= pipeline_zero123.unet.down_blocks[0].resnets[0]

    pipeline_svd.unet.down_blocks[0].resnets[1].spatial_res_block= pipeline_zero123.unet.down_blocks[0].resnets[1]
     

    ############################
    #  CrossAttnDownBlockSpatioTemporal <- CrossAttnDownBlock2D



    #  #attentions
    pipeline_svd.unet.down_blocks[1].attentions[0].transformer_blocks= pipeline_zero123.unet.down_blocks[1].attentions[0].transformer_blocks
    
    pipeline_svd.unet.down_blocks[1].attentions[1].transformer_blocks = pipeline_zero123.unet.down_blocks[1].attentions[1].transformer_blocks

    # resnets
    pipeline_svd.unet.down_blocks[1].resnets[0].spatial_res_block= pipeline_zero123.unet.down_blocks[1].resnets[0]

    pipeline_svd.unet.down_blocks[1].resnets[1].spatial_res_block= pipeline_zero123.unet.down_blocks[1].resnets[1]


    ############################
    #  CrossAttnDownBlockSpatioTemporal <- CrossAttnDownBlock2D



    #  #attentions
    pipeline_svd.unet.down_blocks[2].attentions[0].transformer_blocks= pipeline_zero123.unet.down_blocks[2].attentions[0].transformer_blocks
    
    pipeline_svd.unet.down_blocks[2].attentions[1].transformer_blocks = pipeline_zero123.unet.down_blocks[2].attentions[1].transformer_blocks

    # resnets
    pipeline_svd.unet.down_blocks[2].resnets[0].spatial_res_block= pipeline_zero123.unet.down_blocks[2].resnets[0]

    pipeline_svd.unet.down_blocks[2].resnets[1].spatial_res_block= pipeline_zero123.unet.down_blocks[2].resnets[1]
     

     ############################
    #  DownBlockSpatioTemporal <- DownBlock2D

    #resnets
    pipeline_svd.unet.down_blocks[3].resnets[0].spatial_res_block = pipeline_zero123.unet.down_blocks[3].resnets[0]
    pipeline_svd.unet.down_blocks[3].resnets[1].spatial_res_block = pipeline_zero123.unet.down_blocks[3].resnets[1]



    ###############################################################################
    #upblocks

    ####################################
     #    CrossAttnUpBlockSpatioTemporal <- CrossAttnUpBlock2D

    

    #attentions
    pipeline_svd.unet.up_blocks[3].attentions[0].transformer_blocks= pipeline_zero123.unet.up_blocks[3].attentions[0].transformer_blocks
    
    pipeline_svd.unet.up_blocks[3].attentions[1].transformer_blocks = pipeline_zero123.unet.up_blocks[3].attentions[1].transformer_blocks

    pipeline_svd.unet.up_blocks[3].attentions[2].transformer_blocks = pipeline_zero123.unet.up_blocks[3].attentions[2].transformer_blocks


    # resnets
    pipeline_svd.unet.up_blocks[3].resnets[0].spatial_res_block= pipeline_zero123.unet.up_blocks[3].resnets[0]

    pipeline_svd.unet.up_blocks[3].resnets[1].spatial_res_block= pipeline_zero123.unet.up_blocks[3].resnets[1]

    pipeline_svd.unet.up_blocks[3].resnets[2].spatial_res_block= pipeline_zero123.unet.up_blocks[3].resnets[2]
     
    
    ############################
    #  CrossAttnUpBlockSpatioTemporal <- CrossAttnUpBlock2D



    #  #attentions
    pipeline_svd.unet.up_blocks[2].attentions[0].transformer_blocks= pipeline_zero123.unet.up_blocks[2].attentions[0].transformer_blocks
    
    pipeline_svd.unet.up_blocks[2].attentions[1].transformer_blocks = pipeline_zero123.unet.up_blocks[2].attentions[1].transformer_blocks

    pipeline_svd.unet.up_blocks[2].attentions[2].transformer_blocks = pipeline_zero123.unet.up_blocks[2].attentions[2].transformer_blocks

    # resnets
    pipeline_svd.unet.up_blocks[2].resnets[0].spatial_res_block= pipeline_zero123.unet.up_blocks[2].resnets[0]

    pipeline_svd.unet.up_blocks[2].resnets[1].spatial_res_block= pipeline_zero123.unet.up_blocks[2].resnets[1]

    pipeline_svd.unet.up_blocks[2].resnets[2].spatial_res_block= pipeline_zero123.unet.up_blocks[2].resnets[2]


    ############################
    #  CrossAttnUpBlockSpatioTemporal <- CrossAttnUpBlock2D



    #  #attentions
    pipeline_svd.unet.up_blocks[1].attentions[0].transformer_blocks= pipeline_zero123.unet.up_blocks[1].attentions[0].transformer_blocks
    
    pipeline_svd.unet.up_blocks[1].attentions[1].transformer_blocks = pipeline_zero123.unet.up_blocks[1].attentions[1].transformer_blocks

    pipeline_svd.unet.up_blocks[1].attentions[2].transformer_blocks = pipeline_zero123.unet.up_blocks[1].attentions[2].transformer_blocks


    # resnets
    pipeline_svd.unet.up_blocks[1].resnets[0].spatial_res_block= pipeline_zero123.unet.up_blocks[1].resnets[0]

    pipeline_svd.unet.up_blocks[1].resnets[1].spatial_res_block= pipeline_zero123.unet.up_blocks[1].resnets[1]

    pipeline_svd.unet.up_blocks[1].resnets[2].spatial_res_block= pipeline_zero123.unet.up_blocks[1].resnets[2]
     

     ############################
    #  UpBlockSpatioTemporal <- UpBlock2D

    #resnets
    pipeline_svd.unet.up_blocks[0].resnets[0].spatial_res_block = pipeline_zero123.unet.up_blocks[0].resnets[0]
    pipeline_svd.unet.up_blocks[0].resnets[1].spatial_res_block = pipeline_zero123.unet.up_blocks[0].resnets[1]
    pipeline_svd.unet.up_blocks[0].resnets[2].spatial_res_block = pipeline_zero123.unet.up_blocks[0].resnets[2]


    


pipeline_zero123 = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)

pipeline_svd = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)

merge_unets(pipeline_svd, pipeline_zero123)
# test_svd_unet(pipeline_svd.unet)
pipe = pipeline_svd

pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
# image = Image.open("/localhome/aaa324/Generative Models/VideoPAPR/data/horese.jpg")
image = load_image(image)
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, num_inference_steps= 1).frames[0]

export_to_video(frames, "generated.mp4", fps=7)