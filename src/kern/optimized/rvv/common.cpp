#include "kern/optimized/rv64/common.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace inferllm {
namespace opt {

size_t vlmax[4][4];
size_t vlmul[4][4];

void init() {
    int a;
    asm volatile("csrr %0, vlenb" : "=r"(a));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vlmax[i][j] = a / (1 << i) * (1 << j);
            vlmul[i][j] = j;
        }
    }
}

void dumpCSR() {
#define CSR(a)                                                                      \
    do {                                                                            \
        __asm__ __volatile__("csrr %0, " #a : "=r"(a));                             \
        if (!strcmp(#a, "vtype")) {                                                 \
            printf("vtype(1.0) vill=%lu,vsew=%lu,vlmul=%lu,vta=%lu,vma=%lu\n",      \
                   1 & (a >> 63), 7 & (a >> 2), 3 & a, 1 & (a >> 5), 1 & (a >> 6)); \
            printf("vtype(0.9) vill=%lu,vsew=%lu,vlmul=%lu,vta=%lu,vma=%lu\n",      \
                   1 & (a >> 63), 7 & (a >> 2), 3 & a, 1 & (a >> 5), 1 & (a >> 6)); \
        } else                                                                      \
            printf("%s = 0x%lx\n", #a, a);                                          \
    } while (0);

    size_t vtype, vl, vlenb, vstart, vxrm, vxsat, vcsr;
    CSR(vtype);
    CSR(vl);
    CSR(vlenb);
    CSR(vstart);
    CSR(vxrm);
    CSR(vxsat);
}

}  // namespace opt
}  // namespace inferllm
