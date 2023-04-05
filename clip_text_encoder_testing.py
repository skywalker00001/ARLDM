# import CLIPTokenizer
from transformers import CLIPTokenizer, CLIPTextModel
import torch

clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")

with open('/home/houyi/projects/ARLDM/inf_02_sample/story_00000/story_00000.txt', 'r') as file:
    texts = file.readlines()

texts = [txt.strip() for txt in texts]

#max_length = 79
clip_tokenized = clip_tokenizer(
    texts,
    padding="max_length",
    #max_length=max_length,
    truncation=False,
    return_tensors="pt",
)
# clip_text_model.text_model.embeddings.token_embedding(clip_tokenized["input_ids"]) : torch.Size([4, 77, 768])
# clip_text_model.text_model.embeddings(clip_tokenized["input_ids"]).shape: torch.Size([4, 77, 768])
# clip_text_model_output.hidden_states[0].shape: torch.Size([4, 77, 768])



# torch.equal(clip_text_model.text_model.embeddings(clip_tokenized["input_ids"]), clip_text_model_output.hidden_states[0]): True
clip_text_model = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="text_encoder")


embedding_o = clip_text_model.text_model.embeddings(clip_tokenized["input_ids"])
token_embedding = clip_text_model.text_model.embeddings.token_embedding(clip_tokenized["input_ids"])

added_embedding = clip_text_model.text_model.embeddings.position_embedding(torch.tensor(range(0,77), dtype=torch.int)) + clip_text_model.text_model.embeddings.token_embedding(clip_tokenized["input_ids"])

# torch.equal(added_embedding, embedding_o): True

with torch.inference_mode():
    clip_text_model_output = clip_text_model(output_hidden_states=True, **clip_tokenized)
    # clip_text_model_output.last_hidden_state.shape: torch.Size([4, 77, 768])
    None

None

