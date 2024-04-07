# Convert a chatglm model checkpoint to a InferLLM compatible file
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
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.

import sys
import json
import struct
from enum import Enum
import numpy as np
import torch
import argparse
import tempfile 
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentencepiece import SentencePieceProcessor 

# parse arguments
parser = argparse.ArgumentParser(description="Convert a ChatGLM model to a InferLLM compatible fp16 data type file")
parser.add_argument("-o", "--outfile", type=str, help="the output file")
parser.add_argument("-v", "--version", type=int, default=1, help="the chatglm mode version")
parser.add_argument("-q", "--quantization", type=int, default=32, help="quantization bits")
args = parser.parse_args()

# output in the same directory as the model
model_out_path = args.outfile

class GGMLType(Enum):
    # src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L32
    F32 = 0
    F16 = 1
    QInt4 = 2
    # QUInt4 = 3
    QInt8 = 4

alignment_size = 32
bits = args.quantization
if bits == 32:
    dtype = GGMLType.F32
elif bits == 16:
    dtype = GGMLType.F16
    raise NotImplementedError(f"kernel not suport bits: {bits}")
elif bits == 8:
    dtype = GGMLType.QInt8
elif bits == 4:
    dtype = GGMLType.QInt4
else:
    raise NotImplementedError(f"Unknown quantization bits: {bits}")

version = args.version
if version == 1:
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float().state_dict()
    auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
elif version == 2:
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).float().state_dict()
    auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
elif version == 3:
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float().state_dict()
    auto_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

_, vocab_file = tempfile.mkstemp()
auto_tokenizer.save_vocabulary(vocab_file)
tokenizer = SentencePieceProcessor(vocab_file)

hparams = {
        "embd_size": config.hidden_size,
        "n_heads": config.num_attention_heads,
        "n_layers": config.num_layers,
}
hparams.update({"vocab_size": tokenizer.vocab_size()})

if version > 1:
    hparams.update({"multi_qeury": 1 if config.multi_query_attention else 0})
    hparams.update({"attention_patition": config.multi_query_group_num})
    hparams.update({"fc_hidden": config.ffn_hidden_size})


print(hparams)


fout = open(model_out_path, "wb")
fout.write(struct.pack("i", 0x0123456)) # magic: inferllm

# the model parameters
param_byte = struct.pack("i", hparams["embd_size"])
param_byte +=struct.pack("i", hparams["n_heads"])
param_byte +=struct.pack("i", hparams["n_layers"])
param_byte +=struct.pack("i", hparams["fc_hidden"])
param_byte +=struct.pack("i", hparams["vocab_size"])
if version > 1:
    param_byte +=struct.pack("i", hparams["multi_qeury"])
    param_byte +=struct.pack("i", hparams["attention_patition"])

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



GGML_QK8_0 = 32
GGML_QK4_0 = 32
GGML_QK4_1 = 32


GGML_MEM_ALIGN = 16

def float32Toint8(tensor):
    oriShape = tensor.shape
    newLastElement = oriShape[-1] * 4
    newShape = oriShape[:-1] + (newLastElement,)
    tensor_bytes = tensor.numpy().tobytes()
    return torch.tensor(np.frombuffer(tensor_bytes, dtype=np.int8)).view(newShape)

def offset(tensor, alignment):
    # 计算tensor所占用的字节数
    num_bytes = tensor.element_size() * tensor.nelement()
    # 计算需要填充的字节数
    padding = (alignment - (num_bytes % alignment)) % alignment
    return num_bytes+padding, padding
def quantize_q8_0(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L51
    """
    # equivalent to ggml_quantize_q8_0 in ggml.c

    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK8_0 == 0
    tensor = tensor.view(-1, GGML_QK8_0)
    scale = tensor.abs().max(dim=-1, keepdim=True).values / ((1 << 7) - 1)
    tensor = (tensor / scale).round().clamp(min=-128, max=127).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor

def quantize_quint4(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L62
    """
    # equivalent to ggml_quantize_q4_0 in ggml.c
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale + 8).round().clamp(min=0, max=15).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor



def quantize_qint4(tensor: torch.Tensor) -> torch.Tensor:
    """
    src: https://github.com/li-plus/chatglm.cpp/blob/04910ce72a5d22087ec6e404dbefd73c1ccf2700/chatglm_cpp/convert.py#L62
    """
    # equivalent to ggml_quantize_q4_0 in ggml.c
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.shape[1] % GGML_QK4_0 == 0
    tensor = tensor.view(-1, GGML_QK4_0)
    abs_max_indices = tensor.abs().max(dim=-1, keepdim=True).indices
    max_values = torch.take_along_dim(tensor, abs_max_indices, dim=-1)
    scale = max_values / -8
    tensor = (tensor / scale).round().clamp(min=-8, max=7).char()
    # compress two int4 weights into an int8
    tensor = tensor[:, :16] | (tensor[:, 16:] << 4).type(torch.int8)
    # add scale into each block
    tensor = torch.cat((float32Toint8(scale.float()), tensor), dim=-1)
    return tensor

def dump_tensor(f, name: str, tensor: torch.Tensor, ggml_type: GGMLType):
    assert tensor.dtype == torch.float32
    shape = tensor.shape

    # skip layers.X.attention.inner_attention.rope.freqs
    if name[-5:] == "freqs" or name[-4:]=="freq":
        return


    if name.endswith("query_key_value.weight") or name.endswith("attention.query_key_value.bias"):
        if version == 1:
            tensor = tensor.reshape(32, 3, -1).transpose(0, 1).reshape(-1, 4096)
    dshape = tensor.shape
    sname = name.encode('utf-8')


    if "layernorm" not in name:
        # tensor data
        if ggml_type == GGMLType.F32:
            tensor = tensor.float()
        elif ggml_type == GGMLType.F16:
            tensor = tensor.half()
        elif ggml_type == GGMLType.QInt8:
            tensor = quantize_q8_0(tensor)
        elif ggml_type == GGMLType.QInt4:
            tensor = quantize_qint4(tensor)
        else:
            raise NotImplementedError(f"Cannot dump tensor of dtype {tensor.dtype}")
    else:
        tensor = tensor.float()
        ggml_type = GGMLType.F32

    n_dims = len(shape)
    print("Processing variable: " + name + " with shape: ", shape, " and type: ", ggml_type.value)
    f.write(struct.pack("iii", n_dims, len(sname), ggml_type.value))
    for i in range(n_dims):
        f.write(struct.pack("i", dshape[i]))
    f.write(sname)
    print("write tensor: ", name, " to file :", f.tell())

    tensor.numpy().tofile(f)
    # align address
    if ggml_type == GGMLType.QInt8 or ggml_type == GGMLType.QInt4:
        length, paddingSize =offset(tensor, alignment_size)
        if paddingSize>0:
            paddingTensor = torch.zeros(paddingSize)
            paddingTensor.numpy().tofile(f)
            print("write paddingTensor: ", name, "paddingSize:", paddingSize," to file :", f.tell())

for k, v in model.items():
    dump_tensor(fout, k, v, dtype)


# I hope this deallocates the memory ..
model = None

fout.close()

print("Done. Output file: " + model_out_path)
print("")
