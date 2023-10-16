# predictions = [
#     [[image1], [image2, image3]],
#     [[image4]],
#     [[image5, image6], [image7, image8, image9]]
# ]

# from PIL import Image
# import torch
# import numpy as np
# from torchvision import transforms
# from transformers import CLIPTokenizer


# # blip_image_processor = transforms.Compose([
# #             transforms.Resize([224, 224]),
# #             transforms.ToTensor(),
# #             transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
# #         ])

# # t = blip_image_processor(
# #             Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()

# # print(t.shape)
# clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
# clip_text_null_token = clip_tokenizer([""], padding="max_length", max_length=80,
#                                                    return_tensors="pt").input_ids

# print(clip_text_null_token.shape)

# import numpy as np
# import torch
# import PIL
# import matplotlib.pyplot as plt

# # B, V, V, S
# B, V, S = 2, 5, 10
# square_mask = torch.triu(torch.ones((V, V))).bool()
# square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
# square_mask = square_mask.reshape(B * V, V * S)
# #attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])
# plt.imshow(square_mask)
# plt.savefig("test0.png")
# None
# import inspect

# def foo(a, b, x='blah'):
#     pass

# t = inspect.signature(foo)
# print(inspect.signature(foo))
# # (a, b, x='blah')
# None


# from diffusers import UNet2DConditionModel
# unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
# None
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.tensor([7, 8, 9])

stacked_tensors = torch.stack([a, b, c], dim=0)
print(stacked_tensors)

a+=1
print(stacked_tensors)