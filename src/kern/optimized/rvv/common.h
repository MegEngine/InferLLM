#pragma once

#include <assert.h>
#include <cstdlib>
#include "core/tensor.h"
#include "file.h"
#include "kern/kernel_define.h"

namespace inferllm {
namespace opt {

typedef enum SEW {
    E8 = 0,
    E16 = 1,
    E32 = 2,
    E64 = 3,
} SEW;

extern size_t vlmax[4][4];
extern size_t vlmul[4][4];

inline size_t mk_lmul(SEW sew, int sz) {
    size_t rvlmul;
    for (int i = 0; i < 4; i++) {
        rvlmul = vlmul[sew][i];
        if (sz <= vlmax[sew][i])
            break;
    }
    return rvlmul;
}

inline size_t mk_vtype(SEW sew, size_t LMUL) {
#if INFER_RVV > 107
    return (sew << 3) | LMUL;
#else
    return (sew << 2) | LMUL;
#endif
}

#if INFER_RVV > 107
#define VSET1(e, m) asm volatile("vsetivli x0, 1, " #e ", " #m);
#else
#define VSET1(e, m) asm volatile("vsetvli x0, %[sz], " #e ", " #m ::[sz] "r"(1));
#endif

inline float vmaxabs(size_t sz, const float* t, float init) {
    VSET1(e32, m1);
    asm volatile("vfmv.s.f v1, %[init]\n" : : [init] "f"(init));
    size_t lmul = mk_lmul(E32, sz);
    size_t vt8 = mk_vtype(E8, lmul), vt32 = mk_vtype(E32, lmul);
    for (; sz > 0;) {
        int vl;
        asm volatile(
                "vsetvl        x0, %[sz8], %[vt8]\n"
                "vlbu.v        v8, (%[t])\n"
                "vsetvl        %[vl], %[sz32], %[vt32]\n"
                "vfsgnjx.vv    v8, v8, v8\n"
                "vfredmax.vs   v1, v8, v1\n"
                : [vl] "=r"(vl)
                : [sz8] "r"(sz * 4), [vt8] "r"(vt8), [sz32] "r"(sz), [vt32] "r"(vt32),
                  [t] "r"(t)
                : "memory");
        t += vl;
        sz -= vl;
    }
    VSET1(e32, m1);
    asm volatile("vfmv.f.s  %[init],  v1\n" : [init] "=f"(init));
    return init;
}

inline float vmax(int sz, const float* t, float init) {
    VSET1(e32, m1);
    asm volatile("vfmv.s.f v1, %[init]\n" : : [init] "f"(init));
    size_t lmul = mk_lmul(E32, sz);
    size_t vt8 = mk_vtype(E8, lmul), vt32 = mk_vtype(E32, lmul);
    for (; sz > 0;) {
        int vl;
        asm volatile(
                "vsetvl        x0, %[sz8], %[vt8]\n"
                "vlbu.v        v8, (%[t])\n"
                "vsetvl        %[vl], %[sz32], %[vt32]\n"
                "vfredmax.vs   v1, v8, v1\n"
                : [vl] "=r"(vl)
                : [sz8] "r"(sz * 4), [vt8] "r"(vt8), [sz32] "r"(sz), [vt32] "r"(vt32),
                  [t] "r"(t)
                : "memory");
        t += vl;
        sz -= vl;
    }
    VSET1(e32, m1);
    asm volatile("vfmv.f.s  %[init],  v1\n" : [init] "=f"(init));
    return init;
}

inline float vmulsum(const float* x, const float* y, int sz, float init) {
    asm volatile(
            "vsetvli x0, %[sz], e32, m1\n"
            "vfmv.s.f v1, %[init]\n"
            :
            : [init] "f"(init), [sz] "r"(sz));
    for (; sz > 0;) {
        int vl = 0;
        asm volatile(
                "slli          t0, %[sz], 2\n"
                "vsetvli       t0, t0, e8, m8\n"
                "vlbu.v        v8,  (%[x])\n"
                "vlbu.v        v16, (%[y])\n"
                "srli          t0, t0, 2\n"
                "vsetvli       %[vl], t0, e32, m8\n"
                "vfmul.vv      v8, v8, v16\n"
                "vfredsum.vs   v1, v8, v1\n"
                : [vl] "=r"(vl)
                : [sz] "r"(sz), [x] "r"(x), [y] "r"(y)
                : "t0", "memory");
        x += vl;
        y += vl;
        sz -= vl;
    }
    asm volatile(
            "vsetvli x0, x0, e32, m1\n"
            "vfmv.f.s  %[init],  v1\n"
            : [init] "=f"(init));
    return init;
}

inline float vsqrsum(const float* t, int sz, float init) {
    asm volatile(
            "vsetvli x0, %[sz], e32, m1\n"
            "vfmv.s.f v1, %[init]\n"
            :
            : [init] "f"(init), [sz] "r"(sz));
    for (; sz > 0;) {
        int vl = 0;
        asm volatile(
                "slli          t0, %[sz], 2\n"
                "vsetvli       t0, t0, e8, m8\n"
                "vlbu.v        v8, (%[t])\n"
                "srli          t0, t0, 2\n"
                "vsetvli       %[vl], t0, e32, m8\n"
                "vfmul.vv      v8, v8, v8\n"
                "vfredsum.vs   v1, v8, v1\n"
                : [vl] "=r"(vl)
                : [sz] "r"(sz), [t] "r"(t)
                : "t0", "memory");
        t += vl;
        sz -= vl;
    }
    asm volatile(
            "vsetvli x0, x0, e32, m1\n"
            "vfmv.f.s  %[init],  v1\n"
            : [init] "=f"(init));
    return init;
}

inline void vscal(int sz, const float* x, float* z, float scale) {
    for (; sz > 0;) {
        int vl = 0;
        asm volatile(
                "slli          t0, %[sz], 2\n"
                "vsetvli       t0, t0, e8, m8\n"
                "vlbu.v        v8, (%[x])\n"
                "srli          t1, t0, 2\n"
                "vsetvli       %[vl], t1, e32, m8\n"
                "vfmul.vf      v8, v8, %[scale]\n"
                "vsetvli       x0, t0, e8, m8\n"
                "vsb.v         v8, (%[z])\n"
                : [vl] "=r"(vl)
                : [sz] "r"(sz), [x] "r"(x), [z] "r"(z), [scale] "f"(scale)
                : "t0", "t1", "memory");
        x += vl;
        z += vl;
        sz -= vl;
    }
}

inline void vadd(int sz, const float* x, const float* y, float* z) {
    for (; sz > 0;) {
        int vl = 0;
        asm volatile(
                "slli          t0, %[sz], 2\n"
                "vsetvli       t0, t0, e8, m8\n"
                "vlbu.v        v8, (%[x])\n"
                "vlbu.v        v16, (%[y])\n"
                "srli          t1, t0, 2\n"
                "vsetvli       %[vl], t1, e32, m8\n"
                "vfadd.vv      v8, v8, v16\n"
                "vsetvli       x0, t0, e8, m8\n"
                "vsb.v         v8, (%[z])\n"
                : [vl] "=r"(vl)
                : [sz] "r"(sz), [x] "r"(x), [y] "r"(y), [z] "r"(z)
                : "t0", "t1", "memory");
        x += vl;
        y += vl;
        z += vl;
        sz -= vl;
    }
}

inline void vmul(int sz, const float* x, const float* y, float* z) {
    for (; sz > 0;) {
        int vl = 0;
        asm volatile(
                "slli          t0, %[sz], 2\n"
                "vsetvli       t0, t0, e8, m8\n"
                "vlbu.v        v8, (%[x])\n"
                "vlbu.v        v16, (%[y])\n"
                "srli          t1, t0, 2\n"
                "vsetvli       %[vl], t1, e32, m8\n"
                "vfmul.vv      v8, v8, v16\n"
                "vsetvli       x0, t0, e8, m8\n"
                "vsb.v         v8, (%[z])\n"
                : [vl] "=r"(vl)
                : [sz] "r"(sz), [x] "r"(x), [y] "r"(y), [z] "r"(z)
                : "t0", "t1", "memory");
        x += vl;
        y += vl;
        z += vl;
        sz -= vl;
    }
}

void dumpV();

}  // namespace opt
}  // namespace inferllm
