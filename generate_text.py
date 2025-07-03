import tensorflow as tf
import numpy as np
import argparse
from nanogpt import MultiHeadAttention, FeedForward, GPTModel, TransformerBlock

obj_dict = {
    'MultiHeadAttention': MultiHeadAttention,
    'FeedForward': FeedForward,
    'GPTModel': GPTModel,
    'TransformerBlock': TransformerBlock
}

with open('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)
block_size = 256

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)} 

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = tf.keras.models.load_model('nano_gpt.keras', custom_objects=obj_dict)


def generate_text(txt, max_new_tokens):
  idx = tf.convert_to_tensor([encode(txt)])
  for i in range(max_new_tokens):
      block_idx = idx[:, -block_size:]
      logits, loss = model(block_idx, training=False)
      logits = logits[:, -1, :]
      probas = tf.nn.softmax(logits, axis = -1)
      idx_next = np.argmax(probas, axis = -1, keepdims=True)
      idx = tf.concat([idx, idx_next], axis = -1)

  decoded_text = decode(np.squeeze(idx.numpy()).tolist())
  print(decoded_text)

parser = argparse.ArgumentParser()

parser.add_argument('--start', type=str ,default='helen')
parser.add_argument('--max_tokens', type=int ,default=100)

args = parser.parse_args()


generate_text(txt = args.start, max_new_tokens= args.max_tokens)