# predictions = [
#     [[image1], [image2, image3]],
#     [[image4]],
#     [[image5, image6], [image7, image8, image9]]
# ]

from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from transformers import CLIPTokenizer


# blip_image_processor = transforms.Compose([
#             transforms.Resize([224, 224]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
#         ])

# t = blip_image_processor(
#             Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()

# print(t.shape)
clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
clip_text_null_token = clip_tokenizer([""], padding="max_length", max_length=80,
                                                   return_tensors="pt").input_ids

print(clip_text_null_token.shape)