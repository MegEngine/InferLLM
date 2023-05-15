#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "kern/kernel_define.h"
#include "kern/naive/quantize.h"

using namespace inferllm;
static float table_f32_f16[1 << 16];

#ifdef __ARM_NEON
#include <arm_neon.h>

typedef __fp16 fp16_t;
#define COMPUTE_FP16_TO_FP32(x) ((float)(x))
#else
typedef uint16_t fp16_t;

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float compute_fp16_to_fp32(fp16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || \
        defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value =
            fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value =
            fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff
                                            ? fp32_to_bits(denormalized_value)
                                            : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

#define COMPUTE_FP16_TO_FP32(x) (table_f32_f16[x])
#endif

struct Header {
    int param_offset;
    int param_length;
    int vocab_offset;
    int vocab_length;
    int tensor_offset;
};

struct Param {
    int n_heads;
    int n_layers;
    int embd_dim;
    int fc_hidden;
    int vacab_size;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input model> <output model>\n", argv[0]);
        return 1;
    }
    // init the fp16 lookup table
    fp16_t ii;
    for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = table_f32_f16[i] = COMPUTE_FP16_TO_FP32(ii);
    }

    const std::string inp_model = argv[1];
    const std::string out_model = argv[2];

    printf("%s: loading model from '%s'\n", __func__, inp_model.c_str());

    auto finp = std::ifstream(inp_model, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__,
                inp_model.c_str());
        return false;
    }
    auto fout = std::ofstream(out_model, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__,
                out_model.c_str());
        return false;
    }

    // load model header
    int magic;
    finp.read((char*)&magic, sizeof(magic));
    if (magic != 0x0123456) {
        fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
                inp_model.c_str());
        return false;
    }
    fout.write((char*)&magic, sizeof(magic));

    Header header;
    finp.read((char*)&header, sizeof(header));

    printf("%s: magic         = %d \n", __func__, magic);
    printf("%s: param_offset  = %d \n", __func__, header.param_offset);
    printf("%s: param_length  = %d \n", __func__, header.param_length);
    printf("%s: vocab_offset  = %d \n", __func__, header.vocab_offset);
    printf("%s: vocab_length  = %d \n", __func__, header.vocab_length);
    printf("%s: tensor_offset = %d \n", __func__, header.tensor_offset);
    
    fout.write((char*)&header, sizeof(header));

    // load model params and vocab
    std::vector<char> buf(header.param_length + header.vocab_length);
    finp.read(buf.data(), buf.size());
    fout.write(buf.data(), buf.size());
    fout.seekp(header.tensor_offset, std::ios::beg);

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t> data_u8;
        std::vector<fp16_t> data_f16;
        std::vector<float> data_f32;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char*>(&length), sizeof(length));
            finp.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                finp.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read(&name[0], length);
            {
                static const char* ftype_str[] = {
                        "f32",
                        "f16",
                        "q4_0",
                        "q4_1",
                };
                printf("%s, shape=[%d, %d], type = %s ", name.data(), ne[0],
                       ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                    ".*weight",
            };
            bool quantize = false;
            for (const auto& s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }
            // quantize only 2D tensors
            quantize &= (n_dims == 2);
            //! read tensor data
            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    fprintf(stderr,
                            "%s: unsupported ftype %d for integer "
                            "quantization\n",
                            __func__, ftype);
                    return false;
                }
                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char*>(data_f16.data()),
                              nelements * sizeof(fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = COMPUTE_FP16_TO_FP32(data_f16[i]);
                    }
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char*>(data_f32.data()),
                              nelements * sizeof(float));
                }
                ftype = 2; // quantized to int4
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);
                data_u8.resize(nelements * bpe);
                finp.read(reinterpret_cast<char*>(data_u8.data()),
                          nelements * bpe);
                // if fp16 convert to fp32
                if (ftype == 1) {
                    data_f32.resize(nelements);
                    fp16_t* fp16_data = (fp16_t*)(data_u8.data());
                    for (int i = 0; i < nelements; i++) {
                        data_f32[i] = COMPUTE_FP16_TO_FP32(fp16_data[i]);
                    }
                    data_u8.resize(nelements * sizeof(float));
                    memcpy(data_u8.data(), data_f32.data(),
                           nelements * sizeof(float));
                }
                ftype = 0; // convert to fp32
            }
            // write tensor header
            fout.write(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char*>(&length), sizeof(length));
            fout.write(reinterpret_cast<char*>(&ftype), sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            // quantize tensor data
            if (quantize) {
                work.resize(nelements);  // for quantization

                size_t cur_size = inferllm::naive::quantize_row_q4_0_reference(
                        data_f32.data(), (inferllm::BlockQ40*)work.data(),
                        nelements);

                fout.write(reinterpret_cast<char*>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("quantized length = %zu\n", cur_size);
                printf("quantized from size = %f MB to %f MB \n",
                       nelements * sizeof(fp16_t) / 1024.0 / 1024.0,
                       cur_size / 1024.0 / 1024.0);
            } else {
                printf("not quantized size = %f MB\n",
                       data_u8.size() / 1024.0 / 1024.0);
                fout.write(reinterpret_cast<char*>(data_u8.data()),
                           data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(fp16_t);
        }

        printf("%s: model size  = %8.2f MB\n", __func__,
               total_size_org / 1024.0 / 1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__,
               total_size_new / 1024.0 / 1024.0);
    }

    finp.close();
    fout.close();

    return true;
}