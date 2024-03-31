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

    int32_t n_batch = 8;  // batch size for prompt processing

    std::string model = "ggml-alpaca-7b-q4.bin";  // model path
    std::string prompt;

    bool use_color = true;  // use color to distinguish generations and inputs
    bool use_mmap = false;   // use mmap to load model

    bool interactive = true;        // interactive mode
    bool interactive_start = true;  // reverse prompt immediately
    std::string antiprompt =
            "";  // string upon seeing which more user input is prompted
};

bool app_params_parse(int argc, char** argv, app_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.interactive = false;
            params.interactive_start = false;
            params.use_color = false;

            params.prompt = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            params.interactive = false;
            params.interactive_start = false;
            params.use_color = false;

            std::ifstream file(argv[++i]);

            std::copy(std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>(),
                      back_inserter(params.prompt));

        } else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(argv[++i]);
        } else if (arg == "--top_k") {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx_size") {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--top_p") {
            params.top_p = std::stof(argv[++i]);
        } else if (arg == "--temp") {
            params.temp = std::stof(argv[++i]);
        } else if (arg == "--repeat_last_n") {
            params.repeat_last_n = std::stoi(argv[++i]);
        } else if (arg == "--repeat_penalty") {
            params.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--interactive-start") {
            params.interactive = true;
            params.interactive_start = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            params.antiprompt = argv[++i];
        } else if (arg == "--mmap") {
            params.use_mmap = true;
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
    llm_learning::ModelConfig config;
    config.device = "cpu";
    config.weight_type = "int4";
    config.compt_type = "float32";
    config.nr_thread = params.n_threads;
    config.enable_mmap = params.use_mmap;

    std::shared_ptr<llm_learning::Model> model =
            std::make_shared<llm_learning::Model>(config, "alpaca");
    model->load(params.model);
    model->init(params.top_k, params.top_p, params.temp, params.repeat_penalty,
                params.repeat_last_n, params.seed);

    std::string instruct_inp =
            " Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request.\n\n";
    std::string prompt_inp = "### Instruction:\n\n";
    std::string response_inp = "### Response:\n\n";



    // print the basic parameters
    if (params.interactive) {
        fprintf(stderr, "%s: interactive mode on.\n", __func__);
    }
    fprintf(stderr,
            "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
            "repeat_last_n = %i, repeat_penalty = %f\n",
            params.temp, params.top_k, params.top_p, params.repeat_last_n,
            params.repeat_penalty);
    fprintf(stderr, "\n\n");

    std::vector<char> embd;

    if (params.interactive) {
        fprintf(stderr,
                "== Running in chat mode. ==\n"
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || \
        defined(_WIN32)
                " - Press Ctrl+C to interject at any time.\n"
#endif
                " - If you want to submit another line, end your input in "
                "'\\'.\n");
    }
    // prefill the model with the prompt
    model->prefill(instruct_inp);

    // prompt user immediately after the starting prompt has been loaded
    bool is_interacting = false;
    if (params.interactive_start) {
        is_interacting = true;
    }

    std::string user_input, output;
    //! main loop
    while (model->get_remain_token() > 0) {
        if(!user_input.empty()){
            int token;
            output = model->decode(user_input, token);
            user_input.clear();
            is_interacting = false;
        }
        //! continue to decod to get the next token
        if (!is_interacting) {
            int token;
            output += model->decode_iter(token);
            printf("%s", output.c_str());
            fflush(stdout);

            // token 2 is the end of the instruction
            if (token == 2) {
                printf("\n");
                printf("[end txt]");
                is_interacting = true;
            }
            output.clear();
            // after answering the question, get the user input again
        } else {
            printf("\n> ");
            user_input += prompt_inp;
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
                    buf[n_read] = 0;
                    another_line = true;
                    input.resize(n_read + 1);
                } else {
                    buf[n_read] = '\n';
                    buf[n_read + 1] = 0;
                    another_line = false;
                    input.resize(n_read + 2);
                }
                user_input += input;
            }
            user_input += response_inp;
        }
    }

#if defined(_WIN32)
    signal(SIGINT, SIG_DFL);
#endif
    return 0;
}
