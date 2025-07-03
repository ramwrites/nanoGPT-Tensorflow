# nanoGPT-Tensorflow

This repository contains an implementation of a nanoGPT (Generative Pre-trained Transformer) model built from scratch using Tensorflow and trained on the Tiny Shakespeare dataset. The goal of this project is to create a language model capable of generating text in a style of William Shakespeare’s writings.

This is an autoregressive character-level language model based on the Transformer architecture, designed to generate text one character at a time.

This model has over 10 Million parameters and has been trained for 5000 iterations, with a final validation loss of 1.49.

```
python .\generate_text.py --start "Helen" --max_tokens 300
```
Here’s a sample output generated from the prompt "Helen":
```
Helence the prince of the prince of the world,
Which we shall be so bound to be so both of the world,
Which we shall be so be so born of the storms
That he shall be so be so born to be so.

SICINIUS:
What shall be so?

CORIOLANUS:
I will not so so, sir, I shall be so.

CORIOLANUS:
I will not so so.
```
