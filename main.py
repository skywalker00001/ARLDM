# my implementation, using 8-bit optimizer
# my edited verision 1.0
# author: Hou Yi
# Date: 2023-03-30
# /home/houyi/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5 .   local model

import inspect
import os

import hydra
import datetime
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.diffusers_override.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3
import bitsandbytes as bnb
from collections import defaultdict

# env: arldm3

class LightningDataset(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super(LightningDataset, self).__init__()
        self.kwargs = {"num_workers": args.num_workers, "persistent_workers": True if args.num_workers > 0 else False,
                       "pin_memory": True}
        self.args = args

    def setup(self, stage="fit"):
        if self.args.dataset == "pororo":
            import arldm_datasets.pororo as data
        elif self.args.dataset == 'flintstones':
            import arldm_datasets.flintstones as data
        elif self.args.dataset == 'vistsis':
            import arldm_datasets.vistsis as data
        elif self.args.dataset == 'vistdii':
            import arldm_datasets.vistdii as data
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))
        if stage == "fit":
            self.train_data = data.StoryDataset("train", self.args)
            self.val_data = data.StoryDataset("val", self.args)
        if stage == "test":
            self.test_data = data.StoryDataset("test", self.args)

    def train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)


    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def get_length_of_train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return len(self.trainloader)


class ARLDM(pl.LightningModule):
    """
    ARLDM model
    """
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(ARLDM, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        """
            Configurations
        """
        self.task = args.task

        if args.mode == 'sample':
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True)
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            else:
                raise ValueError("Scheduler not supported")
            self.fid_augment = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        self.blip_image_processor = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.max_length = args.get(args.dataset).max_length

        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()
        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
        
                                                          subfolder="text_encoder")
        
        # len(tokenizer.get_vocab()), 49412, this is a dictionary
        #self.text_encoder.resize_token_embeddings(len(self.clip_tokenizer.get_vocab()))
        self.text_encoder.resize_token_embeddings(args.get(args.dataset).clip_embedding_tokens)
        # resize_position_embeddings
        old_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.max_length)
        self.text_encoder.text_model.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.text_model.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

        self.modal_type_embeddings = bnb.nn.StableEmbedding(2, 768)   # nn.Embedding(2, 768)
        self.time_embeddings = bnb.nn.StableEmbedding(5, 768)
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
            image_size=224, vit='large')
        self.mm_encoder.text_encoder.resize_token_embeddings(args.get(args.dataset).blip_embedding_tokens)

        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)

        self.records = defaultdict(list)

        # Freeze vae and unet
        self.freeze_params(self.vae.parameters())

        if args.freeze_resnet:
            self.freeze_params([p for n, p in self.unet.named_parameters() if "attentions" not in n])

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())
            self.unfreeze_params(self.mm_encoder.text_encoder.embeddings.word_embeddings.parameters())


        # the token embeddings are unfrozen
        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())
            self.unfreeze_params(self.text_encoder.text_model.embeddings.token_embedding.parameters())

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        #optimizer = bnb.optim.Adam8bit(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=self.args.warmup_epochs * self.steps_per_epoch,
                                                  max_epochs=self.args.max_epochs * self.steps_per_epoch)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'step',  # The unit of the scheduler's step size
            }
        }
        return optim_dict

    def forward(self, batch):
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.args.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()
        # images: # test: 5, 3, 218, 218/  train: 5, 3, 512, 512 + normalize
        # captions, attention_mask = clip_tokenized['input_ids'], clip_tokenized['attention_mask']
        # captions shape: B * 4 * 79 (79 is the max_length of the clip_tokenizer )
        # texts: 4 or 5

        images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts = batch
        # B: batch_size, V: 4 or 5 number of story, S: max_length
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        # images: B, V, C, H, W -> B*V, C, H, W (4, 3, 512,, 512)
        images = torch.flatten(images, 0, 1)
        captions = torch.flatten(captions, 0, 1) # (4, 91)
        attention_mask = torch.flatten(attention_mask, 0, 1) # (4, 91)

        source_images = torch.flatten(source_images, 0, 1)  # B* V ,3, 224, 224 (5, 3, 224, 224) 
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1) #(5, 91)
        # 1 is not masked, 0 is masked

        classifier_free_idx = np.random.rand(B * V) < 0.1


        #clip_tokenizer.decode(captions[0], skip_special_tokens=True):  'fred and barney sit on an airplane wing and talk to each other.'
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        # before reshape: torch.Size([5, 91, 768]),  after: 1, 5*91, 768
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        # source_embeddings size? B, 5*77, 768
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0) # B*V, 5*77, 768
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]   # torch.Size([4, 91, 768])
        # source_embeddings: #torch.Size([4, 455, 768])
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)   #torch.Size([4, 455, 768]), (.repeat means 91 * 5)
        # randomly replace a sentence in the story(text and story picture) with null token, this means blanking (1/4) content

        # self.records["caption_embeddings"].append(caption_embeddings.detach().clone())

        # torch.equal(caption_embeddings[2], self.text_encoder(self.clip_text_null_token).last_hidden_state[0]): True


        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))  # torch.Size([768])
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))  # torch.Size([455, 768])
        # 0000011111122222233333334444444, each one has max_length elements
        # torch.Size([4, 546, 768])
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1) # B*V, (1+V) * S, D,  1S for CLIP, V*S for BLIP

        # #attention_mask.shape: torch.Size([4, 546])
        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        
        ## torch.equal(attention_mask[1][91:], attention_mask[2][91:]) :    True

        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        attention_mask[classifier_free_idx] = False



        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])


        # tensor([False, False, False, False, False, False, False, False, False, False,
        # False, False, False, False, False, False, False, False, False, False,
        # False, False, False,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
        #  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,

        # clip_tokenized = clip_tokenizer(
        #     clip_tokenizer.decode(captions[0], skip_special_tokens=True)
        #     padding="max_length",
        #     max_length=self.max_length,
        #     truncation=False,
        #     return_tensors="pt",
        # )

        # recording 
        # record_1 = self.vae.encoder.down_blocks[0].resnets[0].conv1.weight.data
        # record_2 = self.vae.decoder.up_blocks[0].resnets[0].conv2.weight.data
        # record_3 = self.vae.encoder.mid_block.attentions[0].proj_attn.weight.data                        # torch.Size([512, 512])
        # record_4 = self.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[2].weight.data    # torch.Size([320, 1280])
        # record_5 = self.text_encoder.text_model.encoder.layers[0].mlp.fc1.weight.data                    # torch.Size([3072, 768])
        # record_6 = self.mm_encoder.visual_encoder.blocks[0].mlp.fc1.weight.data                          #  torch.Size([4096, 1024])

        # next(self.text_encoder.text_model.embeddings.token_embedding.parameters())
        # next(self.vae.encoder.down_blocks[0].resnets[0].conv1.parameters())
        # next(self.vae.decoder.up_blocks[0].resnets[0].conv2.parameters())
        # next(self.vae.encoder.mid_block.attentions[0].proj_attn.parameters())        # this could have required_grad
        # next(self.unet.down_blocks[0].resnets[0].conv2.parameters())
        # next(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[2].parameters())

        # self.records["record_1"].append(self.vae.encoder.down_blocks[0].resnets[0].conv1.weight.data.detach().clone()) # without required_grad
        # self.records["record_2"].append(self.vae.decoder.up_blocks[0].resnets[0].conv2.weight.data.detach().clone())
        # self.records["record_3"].append(self.vae.encoder.mid_block.attentions[0].proj_attn.weight.data.detach().clone())
        # self.records["record_4"].append(self.unet.down_blocks[0].resnets[0].conv2.weight.data.detach().clone())
        # self.records["record_5"].append(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[2].weight.data.detach().clone())
        # self.records["record_6"].append(self.text_encoder.text_model.encoder.layers[0].mlp.fc1.weight.data.detach().clone())
        # self.records["record_7"].append(self.mm_encoder.visual_encoder.blocks[0].mlp.fc1.weight.data.detach().clone())


        ############################################################################################################################################################################

        # self.records["record_1"].append(next(self.vae.encoder.down_blocks[0].resnets[0].conv1.parameters()).detach().clone())
        # self.records["record_2"].append(next(self.vae.decoder.up_blocks[0].resnets[0].conv2.parameters()).detach().clone())
        # self.records["record_3"].append(next(self.vae.encoder.mid_block.attentions[0].proj_attn.parameters()).detach().clone())
        # self.records["record_4"].append(next(self.unet.down_blocks[0].resnets[0].conv2.parameters()).detach().clone())
        # # print(next(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[2].parameters()))
        # self.records["record_5"].append(next(self.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[2].parameters()).detach().clone())  # false, requires_grad=True
        # self.records["record_6"].append(next(self.text_encoder.text_model.encoder.layers[0].mlp.fc1.parameters()).detach().clone())
        # self.records["record_7"].append(next(self.mm_encoder.visual_encoder.blocks[0].mlp.fc1.parameters()).detach().clone())
        # self.records["record_8"].append(next(self.text_encoder.text_model.embeddings.token_embedding.parameters()).detach().clone())   # false, requires_grad=True

        # self.records["classifier_free_idx"].append(classifier_free_idx)
        # self.records["encoder_hidden_states"].append(encoder_hidden_states.detach().clone())
        # #self.records["caption_embeddings"].append(caption_embeddings)
        # self.records["attention_mask"].append(attention_mask.detach().clone())
        # self.records["clip_null"].append(self.text_encoder(self.clip_text_null_token).last_hidden_state[0].detach().clone()) # self.records["clip_null"][0].shape: torch.Size([91, 768]) , false

        ############################################################################################################################################################################    

        # torch.equal(self.records["record_1"][0], self.records["record_1"][1])
        # torch.equal(self.records["record_2"][0], self.records["record_2"][1])
        # torch.equal(self.records["record_3"][0], self.records["record_3"][1])
        # torch.equal(self.records["record_4"][0], self.records["record_4"][1])
        # torch.equal(self.records["record_5"][0], self.records["record_5"][1])
        # torch.equal(self.records["record_6"][0], self.records["record_6"][1])
        # torch.equal(self.records["classifier_free_idx"][0], self.records["classifier_free_idx"][1])
        # torch.equal(self.records["encoder_hidden_states"][0], self.records["encoder_hidden_states"][1])  # false
        # torch.equal(self.records["caption_embeddings"][0], self.records["caption_embeddings"][1]) # false
        # torch.equal(self.records["attention_mask"][0], self.records["attention_mask"][1]) # false
        # torch.equal(self.records["clip_null"][0], self.records["clip_null"][1]) # True
        # torch.equal(self.records["caption_embeddings"][1][2], self.records["clip_null"][1])

        # torch.equal(self.records["caption_embeddings"][1][2],  self.records["clip_null"][1]): True

        # record the images shape
        latents = self.vae.encode(images).latent_dist.sample() #  latents: torch.Size([4, 4, 64, 64]), images: torch.Size([4, 3, 512, 512])
        latents = latents * 0.1821

        noise = torch.randn(latents.shape, device=self.device)
        bsz = latents.shape[0] # bsz: 4
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=self.device).long() # self.noise_scheduler.num_train_timesteps: 1000
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) # noisy_latents: torch.Size([4,4, 64, 64])

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        # self.unet.down_blocks[0].resnets[0].conv1.weight.data.requires_grad
        #torch.cuda.empty_cache()
        return loss

    def sample(self, batch):

        # test: 5, 3, 218, 218/  train: 5, 3, 512, 512 + normalize

        # original_images.shape: torch.Size([1, 4, 3, 128, 128])
        # captions.shape: torch.Size([1, 4, 91])
        # attention_mask.shape: torch.Size([1, 4, 91])
        # source_images.shape: torch.Size([1, 5, 3, 224, 224])
        # source_caption.shape: torch.Size([1, 5, 91])
        # source_attention_mask.shape: torch.Size([1, 5, 91])
        # text: 4 sentences
        original_images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts = batch

        B, V, S = captions.shape  
        src_V = V + 1 if self.task == 'continuation' else V
        original_images = torch.flatten(original_images, 0, 1) # torch.Size([1, 4, 3, 128, 128])
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)

        if self.task == 'continuation':
            source_images[:, 1:] = 0
        else:
            source_images = 0

        source_images = torch.flatten(source_images, 0, 1) # torch.Size([B*5, 3, 224, 224])
        # if self.task == 'continuation': source_images should be [1, 3, 224, 224], else, should be none.
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D : torch.Size([4, 91, 768])
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)   # ?????? source_images are included?
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)   # torch.Size([4, 455, 768])
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1) # torch.Size([4, 546, 768])

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1) # torch.Size([4, 546])
        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S) # torch.Size([4, 364])
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state  # torch.Size([1, 91, 768])
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1) # torch.Size([1, 455, 768])
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1) # torch.Size([1, 546, 768])
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1) # torch.Size([4, 546, 768])

        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=self.device).bool() # torch.Size([4, 546])
        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        attention_mask = attention_mask.reshape(2, B, V, (src_V + 1) * S) # torch.Size([2, 1, 4, 546])
        images = list() # list of list of PIL image, shape # (B, 512, 512, 3)
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, (src_V + 1) * S, -1) # torch.Size([8, 546, 768]) -> torch.Size([2, 1, 4, 546, 768])
            new_image = self.diffusion(encoder_hidden_states[:, :, i].reshape(2 * B, (src_V + 1) * S, -1), # torch.Size([2, 546, 768]) 
                                       attention_mask[:, :, i].reshape(2 * B, (src_V + 1) * S),  # torch.Size([2, 546])
                                       512, 512, self.args.num_inference_steps, self.args.guidance_scale, 0.0)  # list of PIL image, shape # (B, 512, 512, 3)
            images += new_image

            new_image = torch.stack([self.blip_image_processor(im) for im in new_image]).to(self.device) # torch.Size([1, 3, 224, 224])
            new_embedding = self.mm_encoder(new_image,  # B,C,H,W
                                            source_caption.reshape(B, src_V, S)[:, i + src_V - V], # torch.Size([1, 91])
                                            source_attention_mask.reshape(B, src_V, S)[:, i + src_V - V],
                                            mode='multimodal')  # B, S, D
            # blip_tokenizer.decode(source_caption.reshape(B, src_V, S)[:, i + src_V - V][0]):
            # '[ENC] fred and barney are sitting in a room. fred looks angry and sticks his tongue out while he and barney are talking. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'
            # new_embedding.shape: torch.Size([1, 91, 768])
            new_embedding = new_embedding.repeat_interleave(V, dim=0) # torch.Size([4, 91, 768])
            new_embedding += self.modal_type_embeddings(torch.tensor(1, device=self.device))
            new_embedding += self.time_embeddings(torch.tensor(i + src_V - V, device=self.device))

            encoder_hidden_states = encoder_hidden_states[1].reshape(B * V, (src_V + 1) * S, -1) # torch.Size([4, 546, 768])
            encoder_hidden_states[:, (i + 1 + src_V - V) * S:(i + 2 + src_V - V) * S] = new_embedding
            encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])

        return original_images, images, texts
        # original_images: torch.Size([4, 3, 128, 128]), images 4*PIL.Image (512, 512, 3)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        original_images, images, texts = self.sample(batch) # # original_images: torch.Size([4, 3, 128, 128]), images 4*PIL.Image (512, 512, 3)

        batch_dir = os.path.join(self.args.sample_output_dir, 'story_{:05d}'.format(batch_idx))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)

        tensor_to_pil = transforms.ToPILImage()


        # interpolate to 512*512, the same size as the output of the diffusion model
        resized_original_images = transforms.Resize([512, 512])(original_images)

        ori_gen_save_path = os.path.join(batch_dir, 'ori_gen.png')
        #save_ori_gen_images(tensor_to_pil(resized_original_images), images, ori_gen_save_path)
        
        for i, or_image in enumerate(resized_original_images):
            tensor_to_pil(or_image).save(os.path.join(batch_dir, '{:03d}_GroudTruth.png'.format(i)))

        for i, image in enumerate(images):
            #image.save(os.path.join(args.sample_output_dir, '{:04d}.png'.format(i)))
            image.save(os.path.join(batch_dir, '{:03d}_Generated.png'.format(i)))

        

        all_texts = [text[0] for text in texts]
        all_texts = ("\n").join(all_texts)
        text_file_name = os.path.join( batch_dir ,f"story_{batch_idx:05d}.txt")
        with open(text_file_name, "w") as file:
            file.write(all_texts)

        # if self.args.calculate_fid:
        #     #original_images = original_images.cpu().numpy().astype('uint8') # (4, 3, 128, 128)
        #     # nitian = original_images.cpu().numpy().astype('uint8') 
        #     # nitian = [Image.fromarray(im, 'RGB') for im in nitian] # (4, 3, 128, 128)
        #     original_images = original_images.permute(0, 2, 3, 1).cpu().numpy().astype('uint8') # (4, 3, 128, 128) -> (4, 128, 128, 3)
        #     original_images = [Image.fromarray(im, 'RGB') for im in original_images] # 4 PIL, size = 128 * 3
        #     # [im.save("ori_{}.png".format(idx)) for idx, im in enumerate(original_images)]   # correct
        #     #nini = self.inception_feature(nitian).cpu().numpy()
        #     ori = self.inception_feature(original_images).cpu().numpy()
        #     gen = self.inception_feature(images).cpu().numpy() # 512 * 512
        #     #print() #  calculate_fid_given_features(ori, gen)
        #     # 

        if self.args.calculate_fid:
            #original_images = original_images.cpu().numpy().astype('uint8') # (4, 3, 128, 128)
            original_images = original_images.permute(0, 2, 3, 1).cpu().numpy().astype('uint8') # (4, 3, 128, 128) -> (4, 128, 128, 3)
            original_images = [Image.fromarray(im, 'RGB') for im in original_images] # 4 PIL, size = 128 * 3
            # [im.save("ori_{}.png".format(idx)) for idx, im in enumerate(original_images)]   # correct
            ori = self.inception_feature(original_images).cpu().numpy()
            gen = self.inception_feature(images).cpu().numpy() # 512 * 512

        else:
            ori = None
            gen = None

        return images, original_images,  ori, gen, texts

    def diffusion(self, encoder_hidden_states, attention_mask, height, width, num_inference_steps, guidance_scale, eta):
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                              device=self.device)
        # torch.Size([1, 4, 64, 64])
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps): # tensor([996, 992, 988,....0]), len: 250
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample # torch.Size([2, 4, 64, 64])

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents # torch.Size([1, 4, 64, 64])
        image = self.vae.decode(latents).sample # torch.Size([1, 3, 512, 512])

        image = (image / 2 + 0.5).clamp(0, 1)  
        image = image.cpu().permute(0, 2, 3, 1).numpy()   # (1, 512, 512, 3)

        return self.numpy_to_pil(image)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]

        return pil_images

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    
    @staticmethod
    def save_ori_gen_images(ori, gen, save_path: str):
        None
    # ori: list of Image.
    # gen: list of Image.
        #assert 
        # ori = ori.resize((256, 256))
        # gen = gen.resize((256, 256))
        # grid = Diffusion.image_grid([ori, gen], 1, 2)
        # grid.save(save_path)



    def inception_feature(self, images):
        # transforms.ToTensor()(images[0]) (3, 3, 128)
        images = torch.stack([self.fid_augment(image) for image in images])
        # [transforms.ToPILImage()(self.fid_augment(im)).save("faked_pics/ori_{:02}.png".format(idx)) for idx, im in enumerate(images)]
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False) # torch.Size([4, 3, 299, 299])
        pred = self.inception(images)[0] # torch.Size([4, 2048, 1, 1])

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)


def train(args: DictConfig) -> None:
    dataloader = LightningDataset(args)
    dataloader.setup('fit')
    model = ARLDM(args, steps_per_epoch=dataloader.get_length_of_train_dataloader())


    run_dir = os.path.join(args.ckpt_dir, args.run_name)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(os.path.join(run_dir, 'log')):
        os.makedirs(os.path.join(run_dir, 'log'))

    if args.train_model_file: 
        if not os.path.exists(args.train_model_file):
            os.makedirs(args.train_model_file)

    # logger = TensorBoardLogger(save_dir=run_dir, name='log', default_hp_metric=False)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger = WandbLogger(project='arldm_train_01', name=nowtime, log_model=True, save_dir=run_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        save_top_k=0,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callback_list = [lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=1.0,
        # strategy="ddp",
        # enable_progress_bar=True, progress_bar_refresh_rate=1
        precision=16,  # "16", #"bf16" "mixed"
    )

    
    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def sample(args: DictConfig) -> None:
    assert args.test_model_file is not None, "test_model_file cannot be None"
    # assert args.gpu_ids == 1 or len(args.gpu_ids) == 1, "Only one GPU is supported in test mode"

    if not os.path.exists(args.sample_output_dir):
        try:
            os.mkdir(args.sample_output_dir)
        except:
            pass
    dataloader = LightningDataset(args)
    dataloader.setup('test')
    model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        strategy="ddp",
        # strategy="ddp",
        # precision="16-mixed", #"bf16-mixed"
        # precision=16,
        max_epochs=-1,
        benchmark=True
    )

    predictions = predictor.predict(model, dataloader) # return images, original_images,  ori, gen, texts
    images = [elem for sublist in predictions for elem in sublist[0]]
    original_images = [elem for sublist in predictions for elem in sublist[1]]
    texts = [elem for sublist in predictions for elem in sublist[-1]]

    # the generated step is moved to the predcition_step

    # for i, image in enumerate(images):
    #     #image.save(os.path.join(args.sample_output_dir, '{:04d}.png'.format(i)))
    #     image.save(os.path.join(args.sample_output_dir, '{:04d}_{}.png'.format(i, texts[i])))

    # for i, or_image in enumerate(original_images):
    #     or_image.save(os.path.join(args.sample_output_dir, '{:04d}_GT.png'.format(i)))

    if args.calculate_fid:
        ori = np.array([elem for sublist in predictions for elem in sublist[2]])
        gen = np.array([elem for sublist in predictions for elem in sublist[3]])
        fid = calculate_fid_given_features(ori, gen)
        print('FID: {}'.format(fid))

    text_file_name = os.path.join(args.sample_output_dir ,"FID_score.txt")
    with open(text_file_name, "w") as file:
        file.write('FID: {}'.format(fid))

#@hydra.main(config_path=".", config_name="config")
# @hydra.main(config_path="config", config_name="training_config")
@hydra.main(config_path="config", config_name="inference_config")
def main(args: DictConfig) -> None:
    print(f"Detected devices number: {torch.cuda.device_count()}")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)


if __name__ == '__main__':
    # os.environ["NCCL_DEBUG"] = "INFO"
    main()
