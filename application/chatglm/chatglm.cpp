#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "model.h"

struct app_params {
    int32_t seed = -1;  // RNG seed
    int32_t n_threads =
            std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_predict = 128;     // new tokens to predict
    int32_t repeat_last_n = 64;  // last n tokens to penalize
    int32_t n_ctx = 2048;        // context size

    // sampling parameters
    int32_t top_k = 40;
    float top_p = 0.95f;
    float temp = 0.10f;
    float repeat_penalty = 1.30f;

    std::string model ;  // model path

    bool use_color = true;  // use color to distinguish generations and inputs
    bool use_mmap = false;  // use mmap to load model
    std::string dtype = "float32";  // configure the compute dtype
    std::string mtype = "chatglm";  // the model type name, llama
};

void app_print_usage(int argc, char ** argv, const app_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  --color               colorise output to distinguish prompt and user input from generations\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  --top_k N             top-k sampling (default: %d)\n", params.top_k);
    fprintf(stderr, "  --top_p N             top-p sampling (default: %.1f)\n", params.top_p);
    fprintf(stderr, "  --repeat_last_n N     last n tokens to consider for penalize (default: %d)\n", params.repeat_last_n);
    fprintf(stderr, "  --repeat_penalty N    penalize repeat sequence of tokens (default: %.1f)\n", params.repeat_penalty);
    fprintf(stderr, "  -c N, --ctx_size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  --temp N              temperature (default: %.1f)\n", params.temp);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --mmap                enable mmap when read weights, default = false\n");
    fprintf(stderr, "  -d type               configure the compute type, default float32, can be float32 and flot16 now.\n");
    fprintf(stderr, "  --model_type type     the model type name, default llama, can only be llama now.\n");
    fprintf(stderr, "\n");
}

bool app_params_parse(int argc, char** argv, app_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "-d" || arg == "--dtype") {
            params.dtype = argv[++i];
        } else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--repeat_last_n") {
            params.repeat_last_n = std::stoi(argv[++i]);
        } else if (arg == "--repeat_penalty") {
            params.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mmap") {
            params.use_mmap = true;
        } else if (arg == "-h" || arg == "--help") {
            app_print_usage(argc, argv, params);
            exit(0);
        } else {
            exit(0);
        }
    }

    return true;
}

int main(int argc, char** argv) {
    app_params params;

    if (app_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    int64_t t_load_us = 0;
    inferllm::ModelConfig config;
    config.compt_type = params.dtype;
    config.nr_thread = params.n_threads;
    config.enable_mmap = params.use_mmap;
    config.nr_ctx = params.n_ctx;

    std::shared_ptr<inferllm::Model> model =
            std::make_shared<inferllm::Model>(config, params.mtype);
    model->load(params.model);
    model->init(params.top_k, params.top_p, params.temp, params.repeat_penalty,
                params.repeat_last_n, params.seed);

    // print the basic parameters
    fprintf(stderr, "%s: interactive mode on.\n", __func__);
    fprintf(stderr,
            "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
            "repeat_last_n = %i, repeat_penalty = %f\n",
            params.temp, params.top_k, params.top_p, params.repeat_last_n,
            params.repeat_penalty);
    fprintf(stderr, "\n\n");

    std::vector<char> embd;

    fprintf(stderr,
            "== 运行模型中. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || \
        defined(_WIN32)
            " - 输入 Ctrl+C 将在推出程序.\n"
#endif
            " - 如果你想换行，请在行末输入'\\'符号.\n");

    // prompt user immediately after the starting prompt has been loaded
    auto fix_word = [](std::string& word) {
        auto ret = word;
        if (word == "<n>" || word == "<n><n>")
            word = "\n";
        if (word == "<|tab|>")
            word = "\t";
        int pos = word.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(word.substr(8, word.size() - 10).c_str());
            word = std::string(space_num, ' ');
        }
        pos = word.find("▁");
        if (pos != -1) {
            word.replace(pos, pos + 3, " ");
        }
        // Fix utf-8 garbled characters
        if (word.length() == 6 && word[0] == '<' &&
            word[word.length() - 1] == '>' && word[1] == '0' &&
            word[2] == 'x') {
            int num = std::stoi(word.substr(3, 2), nullptr, 16);
            word = static_cast<char>(num);
        }
    };
    bool is_interacting = true;
    std::string user_input, output;
    //! main loop
    while (model->get_remain_token() > 0) {
        if (!user_input.empty()) {
            int token;
            output = model->decode(user_input, token);
            fix_word(output);
            user_input.clear();
            is_interacting = false;
        }
        //! continue to decod to get the next token
        if (!is_interacting) {
            int token;
            auto o = model->decode_iter(token);
            fix_word(o);
            output += o;
            printf("%s", output.c_str());
            fflush(stdout);

            // token 2 is the end of the instruction
            if (token == 130005) {
                printf("\n");
                is_interacting = true;
            }
            output.clear();
            // after answering the question, get the user input again
        } else {
            printf("\n> ");
            bool another_line = true;
            while (another_line) {
                fflush(stdout);
                std::string input;
                input.resize(256);
                char* buf = const_cast<char*>(input.data());
                int n_read;
                if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
                    // presumable empty line, consume the newline
                    if (scanf("%*c") <= 0) { /*ignore*/
                    }
                    n_read = 0;
                }
                if (n_read > 0 && buf[n_read - 1] == '\\') {
                    buf[n_read - 1] = '\n';
                    another_line = true;
                    input.resize(n_read + 1);
                } else {
                    another_line = false;
                    input.resize(n_read);
                }
                user_input += input;
            }
        }
    }

#if defined(_WIN32)
    signal(SIGINT, SIG_DFL);
#endif
    return 0;
}
