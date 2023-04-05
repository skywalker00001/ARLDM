import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args

        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([512, 512]), # ORIGINAL: 218, 218, 3  
            transforms.ToTensor(), #, now : 3, 512, 512
            transforms.Normalize([0.5], [0.5])
        ])
        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
    

        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("clip {} new tokens added to {}".format(msg, subset))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("blip {} new tokens added to {}".format(msg, subset))

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __getitem__(self, index):   
        # BLIP image: 3, 224, 224
        # CLIP  train image: 3, 512, 512
        # test image: 3, 218, 218,  permuted
        # Fid/ inceptionv3: 3, 64, 64

        if not hasattr(self, 'h5'):
            self.open_h5()

        images = list()
        for i in range(5):
            im = self.h5['image{}'.format(i)][index]
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)  # (640, 128, 3)
            idx = random.randint(0, 4)
            # each image has 5 frames, so we randomly select one
            images.append(im[idx * 128: (idx + 1) * 128])

        source_images = torch.stack([self.blip_image_processor(im) for im in images]) # 3, 224, 224 for each im
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train', 'val'] else torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
        
        # test: 5, 3, 218, 218/  train: 5, 3, 512, 512 + normalize

        texts = self.h5['text'][index].decode('utf-8').split('|')

        # tokenize caption using default tokenizer
        clip_tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = clip_tokenized['input_ids'], clip_tokenized['attention_mask']

        blip_tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = blip_tokenized['input_ids'], blip_tokenized['attention_mask']
        # self.clip_tokenizer.batch_decode(captions, skip_special_tokens=True)
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask, (texts[1:] if self.args.task == 'continuation' else texts)

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        return len(self.h5['text'])
