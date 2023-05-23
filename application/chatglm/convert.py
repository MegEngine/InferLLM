# Convert a ChatGLM model checkpoint to a InferLLM compatible file
#
# Load the model using Torch
# Iterate over all variables and write them to a binary file.
#
# Model Structure Header:
#  - Magic number (int)
#  - Param Offset (int)
#  - Param Length (int)
#  - Vocabulary Offset (int)
#  - Vocabulary Length (int)
#  - Tensor offset (int)
#
# Param :
#  - Hidden Size (int)
#  - Number of heads (int)
#  - Number of layers (int)
#  - Embedding Size (int)
#  - FC hidden size (int)
#  - Vocabulary Size (int)
#  - Weight Data Type (int) (0 = float32, 1 = float16, 2 = int8, 3 = uint8)
#
# For each tensor, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (int8_t[len])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import json
import struct
import numpy as np
import torch
import argparse
import tempfile 
from transformers import AutoTokenizer, AutoModel
from sentencepiece import SentencePieceProcessor 

# parse arguments
parser = argparse.ArgumentParser(description="Convert a ChatGLM model to a InferLLM compatible fp16 data type file")
parser.add_argument("-o", "--outfile", type=str, help="the output file")
args = parser.parse_args()

# output in the same directory as the model
model_out_path = args.outfile

hparams = {
        "embd_size": 4096,
        "n_heads": 32,
        "n_layers": 28,
        "fc_hidden": 16384,
}

model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float().state_dict()
dtype = 0 #1 = float16

auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
_, vocab_file = tempfile.mkstemp()
auto_tokenizer.save_vocabulary(vocab_file)
tokenizer = SentencePieceProcessor(vocab_file)

hparams.update({"vocab_size": tokenizer.vocab_size()})
print(hparams)

fout = open(model_out_path, "wb")
fout.write(struct.pack("i", 0x0123456)) # magic: inferllm

# the model parameters
param_byte = struct.pack("i", hparams["embd_size"])
param_byte +=struct.pack("i", hparams["n_heads"])
param_byte +=struct.pack("i", hparams["n_layers"])
param_byte +=struct.pack("i", hparams["fc_hidden"])
param_byte +=struct.pack("i", hparams["vocab_size"])

# Is this correct??
vocab_byte = bytearray()
for i in range(tokenizer.vocab_size()):
    if tokenizer.is_unknown(i):
        # "<unk>" token (translated as ??)
        text = " \u2047 ".encode("utf-8")
        vocab_byte += struct.pack("i", len(text))
        vocab_byte += text
    elif tokenizer.is_control(i):
        # "<s>"/"</s>" tokens
        vocab_byte += struct.pack("i", 0)
    elif tokenizer.is_byte(i):
        # "<U+XX>" tokens (which may be invalid UTF-8)
        piece = tokenizer.id_to_piece(i)
        if len(piece) != 6:
            print("Invalid token: " + piece)
            sys.exit(1)
        byte_value = int(piece[3:-1], 16)
        vocab_byte += struct.pack("i", 1)
        vocab_byte += struct.pack("B", byte_value)
    else:
        # normal token. Uses U+2581 (LOWER ONE EIGHTH BLOCK) to represent spaces.
        text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
        vocab_byte += struct.pack("i", len(text))
        vocab_byte += text

# write the model header

param_offset_addr = fout.tell() # param offset
fout.seek(4, 1) # skip param offset length
fout.write(struct.pack("i", len(param_byte))) # param length
vocab_offset_addr = fout.tell() # vocab offset
fout.seek(4, 1) # skip vocab offset length
fout.write(struct.pack("i", len(vocab_byte))) # vocab length
tensor_offset_addr = fout.tell() # tensor offset
fout.seek(4, 1) # skip tensor offset length

param_offset = fout.tell()
# write the model parameters
fout.write(param_byte)
vocal_offset = fout.tell()
fout.write(vocab_byte)
tensor_offset = fout.tell()

# write the offsets
fout.seek(param_offset_addr, 0)
fout.write(struct.pack("i", param_offset))
fout.seek(vocab_offset_addr, 0)
fout.write(struct.pack("i", vocal_offset))
fout.seek(tensor_offset_addr, 0)
fout.write(struct.pack("i", tensor_offset))

# seek to the end of the file
fout.seek(0, 2)

for k, v in model.items():
    name = k
    shape = v.shape

    # skip layers.X.attention.inner_attention.rope.freqs
    if name[-5:] == "freqs":
        continue

    print("Processing variable: " + name + " with shape: ", shape, " and type: ", v.dtype)
    if name.endswith("query_key_value.weight") or name.endswith("attention.query_key_value.bias"):
        v = v.reshape(32, 3, -1).transpose(0, 1).reshape(-1, 4096)

    data = v.numpy().squeeze()
    n_dims = len(data.shape)

    dshape = data.shape
    sname = name.encode('utf-8')
    print("write tensor: ", name, " to file :", fout.tell())
    fout.write(struct.pack("iii", n_dims, len(sname), dtype))
    for i in range(n_dims):
        fout.write(struct.pack("i", dshape[i]))
    fout.write(sname)

    # data
    data.tofile(fout)

# I hope this deallocates the memory ..
model = None

fout.close()

print("Done. Output file: " + model_out_path)
print("")