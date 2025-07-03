
""" Cofig """

cfg = {
"block_size" : 256,
"batch_size": 64,
"dropout": 0.2,
"n_embd" : 384,
"n_heads" : 6,
"vocab_size": 65,
"n_layers": 6,
"head_dim": int(384/6),
"learning_rate" : 3e-4,
"max_iters": 5000,
"eval_interval": 500,
"eval_iters": 200
}

""" Importing Libraries """

import numpy as np
import tensorflow as tf
import os

""" MultiHeadAttention """

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, n_embd, name=None):
        super().__init__(name=name)
        self.n_embd = n_embd # 384
        self.num_heads = n_heads  # 6
        self.head_dim = int(n_embd/n_heads) # 384/6 = 64
        self.W_key = tf.keras.layers.Dense(n_embd, use_bias=False)
        self.W_query = tf.keras.layers.Dense(n_embd, use_bias=False)
        self.W_value = tf.keras.layers.Dense(n_embd, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(cfg['dropout'])
        self.tril = tf.linalg.band_part(tf.Variable(tf.ones((cfg['block_size'], cfg['block_size'])),
                                                    trainable=False),
                                        num_lower=-1,
                                        num_upper=0)
        self.out_proj = tf.keras.layers.Dense(n_embd)

    def call(self, x, training = True):
        b, num_tokens, embd_dim = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = tf.reshape(keys, (b, num_tokens, self.num_heads, self.head_dim))
        queries = tf.reshape(queries, (b, num_tokens, self.num_heads, self.head_dim))
        values = tf.reshape(values, (b, num_tokens, self.num_heads, self.head_dim))

        keys = tf.transpose(keys, perm=[0,2,1,3])
        queries = tf.transpose(queries, perm=[0,2,1,3])
        values = tf.transpose(values, perm=[0,2,1,3])

        attention_scores = queries @ tf.transpose(keys, perm=[0, 1, 3, 2]) * keys.shape[-1]**-0.5
        mask = self.tril[:num_tokens, :num_tokens]

        attention_scores = tf.where(mask == 0, tf.fill(attention_scores.shape, float('-inf')), attention_scores)

        attention_weights = tf.nn.softmax(attention_scores, axis = -1)
        attention_weights = self.dropout(attention_weights, training = training)

        context_vec = tf.transpose(attention_weights @ values, perm = [0,2,1,3])

        context_vec = tf.reshape(context_vec, (b, num_tokens, self.n_embd))
        context_vec = self.out_proj(context_vec)
        return context_vec

""" Feed Forward """

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg['n_embd']*4),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(cfg['n_embd']),
            ]
        )
    def call(self, x, training = True):
        return self.net(x)

""" Transformer Block """

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads=cfg['n_heads'], n_embd=cfg['n_embd'])
        self.feed = FeedForward(cfg)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.drop_shortcut = tf.keras.layers.Dropout(cfg['dropout'])
    def call(self, x, training = True):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.drop_shortcut(x, training = training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
""" Loss """

def estimate_loss(model):
    out = []
    training = False
    for split in ['train', 'val']:
        losses = []
        for k in range(cfg['eval_iters']):
            x, y = get_batch(split)
            logits, loss = model(x,y, training = training)
            losses.append(loss.numpy())
        out.append(tf.reduce_mean(losses))
    return tf.convert_to_tensor(out)

def loss_func(targets, logits):
    loss_values = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    return tf.reduce_mean(loss_values)

""" GPT Class"""

class GPTModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global cfg
        self.tok_emb = tf.keras.layers.Embedding(input_dim=cfg['vocab_size'], output_dim=cfg['n_embd'])
        self.pos_emb = tf.keras.layers.Embedding(input_dim=cfg['block_size'], output_dim=cfg['n_embd'])
        self.drop = tf.keras.layers.Dropout(cfg['dropout'])
        self.trf_blocks = tf.keras.Sequential(
            [TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.out_head = tf.keras.layers.Dense(cfg['vocab_size'], use_bias=False)
    def call(self, in_idx, targets=None, training = True):
        batch, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(tf.range(seq_len))
        x = tok_emb + pos_emb
        x = self.drop(x, training = training)
        x = self.trf_blocks(x, training = training)
        x = self.final_norm(x)
        logits = self.out_head(x) # (b, num_tokens, vocab_size)

        if targets == None:
            loss = None
        else:
            loss = loss_func(targets, logits)
        return logits, loss




if __name__ == '__main__':

    """ Importing Text File """

    with open('/content/drive/MyDrive/Colab Notebooks/nano GPT/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))

    vocab_size = len(chars)

    """ TokenID """

    stoi = {ch:i for i, ch in enumerate(chars)} # chars to tokenID mapping
    itos = {i:ch for i, ch in enumerate(chars)} # tokenID to chars mapping

    # given a string it converts every char to tokenID and returns a list
    encode = lambda s: [stoi[c] for c in s]
    # given a list of tokenID's it converts it back to string and returns it
    decode = lambda l: ''.join([itos[i] for i in l])

    """ Train Val Split """

    data = tf.convert_to_tensor(encode(text))

    n = int(0.9* len(data))
    train = data[:n]
    val = data[n:]

    print(len(train))
    print(len(val))

    """ Data Loading """

    def get_batch(split):
        data = train if split == 'train' else val
        # this generates the start index for each batch from [0, len-blocksize]
        ix = np.random.randint(low = 0, high=len(data)-cfg['block_size'], size = cfg['batch_size'])
        # this creates the x and y from the start index
        x = tf.stack([data[i: i+cfg['block_size']] for i in ix], axis = 0)
        y = tf.stack([data[i+1: i+cfg['block_size']+1] for i in ix], axis = 0)

        return x, y

    x, y = get_batch('train')

    """ Generate Text """

    def generate_text(model, txt, max_new_tokens):
        idx = tf.convert_to_tensor([encode(txt)])
        for i in range(max_new_tokens):
            block_idx = idx[:, -cfg['block_size']:]
            logits, loss = model(block_idx, training=False)
            logits = logits[:, -1, :]
            probas = tf.nn.softmax(logits, axis = -1)
            idx_next = np.argmax(probas, axis = -1, keepdims=True)
            idx = tf.concat([idx, idx_next], axis = -1)

        decoded_text = decode(np.squeeze(idx.numpy()).tolist())
        print(decoded_text)

    nano_gpt = GPTModel(cfg)

    """ Training """

    opt = tf.keras.optimizers.AdamW(learning_rate=cfg['learning_rate'])

    build = nano_gpt(x, y)

    folder_path = '/content/drive/MyDrive/Colab Notebooks/nano GPT'

    for i in range(cfg['max_iters']):
        xb, yb = get_batch('train')
        with tf.GradientTape() as tape:
            logits, loss = nano_gpt(xb, yb)
        gradients = tape.gradient(loss, nano_gpt.trainable_variables)
        opt.apply_gradients(zip(gradients, nano_gpt.trainable_variables))
        if i%cfg['eval_interval'] == 0:
            losses = tf.stop_gradient(estimate_loss(nano_gpt))
            print(f'train_loss: {losses[0]} and val_loss: {losses[1]}')
            nano_gpt.save(os.path.join(folder_path, f'model{i}.keras'))

    nano_gpt.save(os.path.join(folder_path, 'nano_gpt.keras'))

    estimate_loss(nano_gpt)

    generate_text(nano_gpt, 'helen', 700)

