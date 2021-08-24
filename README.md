# Transformer
A translator which can translate english sentence to chinese sentence, and it is made by using architecture of Transformer ( https://arxiv.org/abs/1706.03762 ).

## File tree

```
.
├── Transformer_tutorial.ipynb
├── __pycache__
│   └── transformer.cpython-38.pyc
├── main.py
├── nmt
│   ├── checkpoints
│   ├── en_vocab.subwords
│   ├── logs
│   └── zh_vocab.subwords
├── tensorflow-datasets
│   └── downloads
└── transformer.py
```

## Usage

### Set hyper parameters
#### -- set
- Default:
  - num_layers = 4
  - d_model = 128
  - dff = 512
  - num_heads = 8

```sh
python3 main.py --set 4 128 512 8
```
### Train your own model
#### -- train
- Default:
  - EPOCH = 30
```sh
python3 main.py --train 30
```

### Make a translation
#### --zh
```sh
python3 main.py --zh China, India, and others have enjoyed continuing economic growth.
```
