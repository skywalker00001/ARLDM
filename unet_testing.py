from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet")
named_p = unet.named_parameters()


for n, p in named_p:
    print(n, p.shape)
    None

None



#unet
# unet2 = UNet2DConditionModel(
#   (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (time_proj): Timesteps()
#   (time_embedding): TimestepEmbedding(
#     (linear_1): Linear(in_features=320, out_features=1280, bias=True)
#     (act): SiLU()
#     (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
#   )
#   (down_blocks): ModuleList(
#     (0): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (3): DownBlock2D(
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#     )
#   )
#   (up_blocks): ModuleList(
#     (0): UpBlock2D(
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (3): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (attn2): CrossAttention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#     )
#   )
#   (mid_block): UNetMidBlock2DCrossAttn(
#     (attentions): ModuleList(
#       (0): Transformer2DModel(
#         (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#         (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         (transformer_blocks): ModuleList(
#           (0): BasicTransformerBlock(
#             (attn1): CrossAttention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (ff): FeedForward(
#               (net): ModuleList(
#                 (0): GEGLU(
#                   (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                 )
#                 (1): Dropout(p=0.0, inplace=False)
#                 (2): Linear(in_features=5120, out_features=1280, bias=True)
#               )
#             )
#             (attn2): CrossAttention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=768, out_features=1280, bias=False)
#               (to_v): Linear(in_features=768, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           )
#         )
#         (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (resnets): ModuleList(
#       (0): ResnetBlock2D(
#         (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#       (1): ResnetBlock2D(
#         (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#     )
#   )
#   (conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
#   (conv_act): SiLU()
#   (conv_out): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )
# sum(p.numel() for p in unet.parameters())
# 859520964