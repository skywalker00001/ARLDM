# deal with the embedding

import torch
from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")


max_length = 79
old_embeddings = text_encoder.text_model.embeddings.position_embedding
new_embeddings = text_encoder._get_resized_embeddings(old_embeddings, max_length)
text_encoder.text_model.embeddings.position_embedding = new_embeddings
text_encoder.config.max_position_embeddings = max_length
text_encoder.max_position_embeddings = max_length
text_encoder.text_model.embeddings.position_ids = torch.arange(max_length).expand((1, -1))

# tokenizer.vocab_size    49408

# text_encoder.get_input_embeddings()   Embedding(49408, 768)    text_encoder.get_input_embeddings().weight.data.shape

new_tokens = [ "fred", "barney", "wilma", "betty", "pebbles", "dino", "slate" ]
# {'barney': 49408, 'wilma': 49409, 'pebbles': 49410, 'slate': 49411}
t = tokenizer.add_tokens(new_tokens)

# len(tokenizer.get_vocab()), 49412, this is a dictionary

# resize the model's embedding layer
# num_added_tokens = len(new_tokens)

ori_embedding = text_encoder.get_input_embeddings().weight.data

text_encoder.resize_token_embeddings(len(tokenizer.get_vocab()))

new_embedding = text_encoder.get_input_embeddings().weight.data


# verify that the new tokens have been added to the vocabulary
print(tokenizer.get_vocab()['my_token_1'])   # output: <num_tokens_in_vocab + 1>
print(tokenizer.get_vocab()['my_token_2'])   # output: <num_tokens_in_vocab + 2>
None


# torch.equal((text_encoder._get_resized_embeddings(text_encoder.text_model.embeddings.position_embedding, 79).weight.data)[0:78], text_encoder.text_model.embeddings.position_embedding.weight.data[0:78])


