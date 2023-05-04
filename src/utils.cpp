#include "utils.h"

#include <math.h>
#include <stdarg.h>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

namespace inferllm{

 #if defined(_MSC_VER) || defined(__MINGW32__)
 #include <malloc.h> // using malloc.h with MSC/MINGW
 #elif !defined(__FreeBSD__) && !defined(__NetBSD__)
 #include <alloca.h>
 #endif

void sample_top_k(std::vector<std::pair<double, Vocab::Id>>& logits_id,
                  int top_k) {
    // find the top K tokens
    std::partial_sort(logits_id.begin(), logits_id.begin() + top_k,
                      logits_id.end(),
                      [](const std::pair<double, Vocab::Id>& a,
                         const std::pair<double, Vocab::Id>& b) {
                          return a.first > b.first;
                      });

    logits_id.resize(top_k);
}

Vocab::Id llama_sample_top_p_top_k(
        const Vocab & vocab,
        const float * logits,
        std::list<Vocab::Id> & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, Vocab::Id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (logits[i] < 0.0) {
                    logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
                }                
            } else {
                logits_id.push_back(std::make_pair(logits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

}  // namespace inferllm
