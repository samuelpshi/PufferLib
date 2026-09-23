#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"

#ifdef __cplusplus
#define BP_STATIC_ASSERT static_assert
#else
#define BP_STATIC_ASSERT _Static_assert
#endif

#ifdef DEBUG
#define BP_CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "INVARIANT %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); abort(); } } while (0)
#else
#define BP_CHECK(cond, ...) ((void)0)
#endif

#define BP_SIZE       10
#define BP_MIN_DIM    2
#define BP_MAX_DIM    (BP_SIZE / 2)
#define BP_MAX_BOXES  125
#define BP_CELLS      (BP_SIZE * BP_SIZE)
#define BP_VOLUME     (BP_SIZE * BP_SIZE * BP_SIZE)
#define BP_NORIENT    6
#define BP_ACT_TOTAL  (BP_NORIENT * BP_CELLS)
#ifndef BP_PREVIEW_K
#define BP_PREVIEW_K  0
#endif

#define BP_SUPPORT_1  60
#define BP_SUPPORT_2  80
#define BP_SUPPORT_3  95

BP_STATIC_ASSERT(BP_MAX_DIM * 2 == BP_SIZE, "MAX_DIM must be SIZE/2");
BP_STATIC_ASSERT(BP_MIN_DIM >= 2, "corner rule L4 needs four distinct corners");
BP_STATIC_ASSERT(BP_MAX_BOXES == (BP_SIZE / BP_MIN_DIM) * (BP_SIZE / BP_MIN_DIM) * (BP_SIZE / BP_MIN_DIM),
                 "MAX_BOXES = (SIZE/MIN_DIM)^3");
BP_STATIC_ASSERT(BP_MAX_BOXES < 256, "voxel owner is uint8");
BP_STATIC_ASSERT(BP_SIZE < 256, "rest_z is uint8");

#define OBS_SIZE  (BP_CELLS + 3 * (1 + BP_PREVIEW_K))
#define ACT_SIZES {BP_ACT_TOTAL}
#define NUM_ATNS  1

enum { BP_POLICY_NONE = 0, BP_POLICY_RANDOM = 1, BP_POLICY_DBL = 2, BP_POLICY_FLAT = 3 };
enum { BP_LEGAL = 0, BP_FAIL_ORIENT, BP_FAIL_FOOTPRINT, BP_FAIL_HEIGHT, BP_FAIL_SUPPORT };

typedef struct {
    uint8_t d[3];
    uint8_t tx, ty, tz;
} BpBox;

typedef struct {
    uint8_t inst;
    uint8_t o;
    uint8_t dx, dy, dz;
    uint8_t x, y, z;
} BpPlaced;

typedef struct {
    uint8_t x, y, z;
    uint8_t l[3];
} BpBlock;

struct Log {
    float perf;
    float score;
    float utilization;
    float boxes_placed;
    float boxes_remaining;
    float mean_legal_actions;
    float mask_cause_orient;
    float mask_cause_footprint;
    float mask_cause_height;
    float mask_cause_support;
    float illegal_actions;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;

    unsigned int gen_rng;
    unsigned int pol_rng;

    int num_orient;
    int stability;
    int policy;
    int eval_seeds;
    int ep_count;

    uint8_t height[BP_CELLS];
    uint8_t voxel[BP_SIZE * BP_CELLS];
    uint8_t rest_z[BP_ACT_TOTAL];

    BpBox instance[BP_MAX_BOXES];
    BpPlaced placed[BP_MAX_BOXES];
    BpBlock cut_stack[BP_MAX_BOXES];

    int n_boxes, n_placed, cursor;
    int vol_placed;

    long legal_sum;
    long cause_sum[5];
    int illegal;
    float ep_return;

    BpPlaced last_placed[BP_MAX_BOXES];
    int last_n, last_boxes, render_hold;
    float last_util;
};

static inline int bp_cell(int x, int y) { return y * BP_SIZE + x; }

static inline int bp_encode(int o, int x, int y) { return o * BP_CELLS + bp_cell(x, y); }

static inline void bp_decode(int a, int *o, int *x, int *y) {
    *o = a / BP_CELLS;
    int c = a % BP_CELLS;
    *y = c / BP_SIZE;
    *x = c % BP_SIZE;
}

static inline int bp_vox(int x, int y, int z) { return z * BP_CELLS + bp_cell(x, y); }

static const uint8_t BP_PERM[BP_NORIENT][3] = {
    {0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {2, 0, 1}, {1, 2, 0}, {2, 1, 0},
};

static inline void bp_orient(const BpBox *b, int o, int *dx, int *dy, int *dz) {
    *dx = b->d[BP_PERM[o][0]];
    *dy = b->d[BP_PERM[o][1]];
    *dz = b->d[BP_PERM[o][2]];
}

static inline unsigned int bp_splitmix32(unsigned int s) {
    s += 0x9e3779b9u;
    s ^= s >> 16; s *= 0x7feb352du;
    s ^= s >> 15; s *= 0x846ca68bu;
    s ^= s >> 16;
    return s == 0 ? 1u : s;
}

static inline unsigned int bp_xorshift(unsigned int *st) {
    unsigned int x = *st;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *st = x;
    return x;
}

static inline int bp_below(unsigned int *st, int n) {
    return (int)(bp_xorshift(st) % (unsigned int)n);
}

static void bp_generate(Env *e, unsigned int seed) {
    e->gen_rng = bp_splitmix32(seed);
    e->n_boxes = 0;

    int top = 0;
    BpBlock root = {0, 0, 0, {BP_SIZE, BP_SIZE, BP_SIZE}};
    e->cut_stack[top++] = root;

    while (top > 0) {
        BP_CHECK(top <= BP_MAX_BOXES, "I-G4 cut stack depth %d", top);
        BpBlock b = e->cut_stack[--top];

        int viol[3], nv = 0;
        for (int k = 0; k < 3; k++) if (b.l[k] > BP_MAX_DIM) viol[nv++] = k;

        if (nv == 0) {
            BP_CHECK(e->n_boxes < BP_MAX_BOXES, "box count overflow");
            BpBox *bx = &e->instance[e->n_boxes++];
            bx->d[0] = b.l[0]; bx->d[1] = b.l[1]; bx->d[2] = b.l[2];
            bx->tx = b.x; bx->ty = b.y; bx->tz = b.z;
            continue;
        }

        int ax = viol[bp_below(&e->gen_rng, nv)];
        int L = b.l[ax];
        int p = BP_MIN_DIM + bp_below(&e->gen_rng, L - 2 * BP_MIN_DIM + 1);

        BpBlock lo = b, hi = b;
        lo.l[ax] = (uint8_t)p;
        hi.l[ax] = (uint8_t)(L - p);
        if (ax == 0) hi.x = (uint8_t)(b.x + p);
        if (ax == 1) hi.y = (uint8_t)(b.y + p);
        if (ax == 2) hi.z = (uint8_t)(b.z + p);
        e->cut_stack[top++] = lo;
        e->cut_stack[top++] = hi;
    }

    int gvol = 0;
    for (int i = 0; i < e->n_boxes; i++)
        gvol += e->instance[i].d[0] * e->instance[i].d[1] * e->instance[i].d[2];
    if (gvol != BP_VOLUME) {
        fprintf(stderr, "binpack: generator volume %d != %d (seed %u)\n", gvol, BP_VOLUME, seed);
        abort();
    }

    unsigned int key[BP_MAX_BOXES];
    for (int i = 0; i < e->n_boxes; i++) key[i] = bp_xorshift(&e->gen_rng);

    for (int i = 1; i < e->n_boxes; i++) {
        BpBox b = e->instance[i];
        unsigned int k = key[i];
        int j = i - 1;
        while (j >= 0 && (e->instance[j].tz > b.tz ||
                          (e->instance[j].tz == b.tz && key[j] > k))) {
            e->instance[j + 1] = e->instance[j];
            key[j + 1] = key[j];
            j--;
        }
        e->instance[j + 1] = b;
        key[j + 1] = k;
    }

#ifdef DEBUG

    uint8_t claim[BP_VOLUME];
    memset(claim, 0, sizeof(claim));
    int vol = 0;
    for (int i = 0; i < e->n_boxes; i++) {
        BpBox *b = &e->instance[i];
        for (int k = 0; k < 3; k++)
            BP_CHECK(b->d[k] >= BP_MIN_DIM && b->d[k] <= BP_MAX_DIM, "I-G1 box %d dim %d = %d", i, k, b->d[k]);
        vol += b->d[0] * b->d[1] * b->d[2];
        for (int z = b->tz; z < b->tz + b->d[2]; z++)
            for (int y = b->ty; y < b->ty + b->d[1]; y++)
                for (int x = b->tx; x < b->tx + b->d[0]; x++) {
                    BP_CHECK(x < BP_SIZE && y < BP_SIZE && z < BP_SIZE, "I-G3 target out of bounds");
                    int v = bp_vox(x, y, z);
                    BP_CHECK(claim[v] == 0, "I-G3 cell (%d,%d,%d) claimed twice", x, y, z);
                    claim[v] = 1;
                }
        if (i > 0) BP_CHECK(e->instance[i - 1].tz <= b->tz, "G6 order");
    }
    BP_CHECK(vol == BP_VOLUME, "I-G2 volume %d", vol);
#endif
}

static inline int bp_rest_z(const Env *e, int x, int y, int dx, int dy) {
    int z = 0;
    for (int j = y; j < y + dy; j++)
        for (int i = x; i < x + dx; i++)
            if (e->height[bp_cell(i, j)] > z) z = e->height[bp_cell(i, j)];
    return z;
}

static inline int bp_stable(const Env *e, int x, int y, int dx, int dy, int z) {
    int area = dx * dy, sup = 0;
    for (int j = y; j < y + dy; j++)
        for (int i = x; i < x + dx; i++)
            sup += (e->height[bp_cell(i, j)] == z);
    int x1 = x + dx - 1, y1 = y + dy - 1;
    int corners = (e->height[bp_cell(x, y)] == z) + (e->height[bp_cell(x1, y)] == z)
                + (e->height[bp_cell(x, y1)] == z) + (e->height[bp_cell(x1, y1)] == z);
    if (sup * 100 > BP_SUPPORT_1 * area && corners == 4) return 1;
    if (sup * 100 > BP_SUPPORT_2 * area && corners >= 3) return 1;
    if (sup * 100 > BP_SUPPORT_3 * area) return 1;
    return 0;
}

static int bp_check(const Env *e, const BpBox *b, int o, int x, int y, int *z_out) {
    if (o >= e->num_orient) return BP_FAIL_ORIENT;
    int dx, dy, dz;
    bp_orient(b, o, &dx, &dy, &dz);
    if (x + dx > BP_SIZE || y + dy > BP_SIZE) return BP_FAIL_FOOTPRINT;
    int z = bp_rest_z(e, x, y, dx, dy);
    if (z + dz > BP_SIZE) return BP_FAIL_HEIGHT;
    if (e->stability && !bp_stable(e, x, y, dx, dy, z)) return BP_FAIL_SUPPORT;
    if (z_out) *z_out = z;
    return BP_LEGAL;
}

static inline int bp_is_legal(const Env *e, int a) {
    int o, x, y;
    bp_decode(a, &o, &x, &y);
    return bp_check(e, &e->instance[e->cursor], o, x, y, NULL) == BP_LEGAL;
}

static int bp_build_mask(Env *e, long causes[5]) {
    unsigned char *mask = e->agents[0].action_mask;
    const BpBox *b = &e->instance[e->cursor];
    for (int k = 0; k < 5; k++) causes[k] = 0;
    int pop = 0;
    for (int a = 0; a < BP_ACT_TOTAL; a++) {
        int o, x, y, z = 0;
        bp_decode(a, &o, &x, &y);
        int r = bp_check(e, b, o, x, y, &z);
        causes[r]++;
        mask[a] = (unsigned char)(r == BP_LEGAL);
        if (r == BP_LEGAL) { e->rest_z[a] = (uint8_t)z; pop++; }
    }
    return pop;
}

static void bp_place(Env *e, int a) {
    int o, x, y, dx, dy, dz;
    bp_decode(a, &o, &x, &y);
    const BpBox *b = &e->instance[e->cursor];
    bp_orient(b, o, &dx, &dy, &dz);
    int z = bp_rest_z(e, x, y, dx, dy);

#ifdef DEBUG
    BP_CHECK(bp_check(e, b, o, x, y, NULL) == BP_LEGAL, "illegal action %d reached bp_place (o%d x%d y%d)", a, o, x, y);
    BP_CHECK(z == e->rest_z[a], "I-S7 rest z %d vs mask %d", z, e->rest_z[a]);
    for (int j = y; j < y + dy; j++)
        for (int i = x; i < x + dx; i++)
            BP_CHECK(e->height[bp_cell(i, j)] <= z, "I-M5 drop path blocked");
#endif

    int id = e->n_placed + 1;
    for (int k = z; k < z + dz; k++)
        for (int j = y; j < y + dy; j++)
            for (int i = x; i < x + dx; i++) {
                BP_CHECK(e->voxel[bp_vox(i, j, k)] == 0, "I-S3 cell (%d,%d,%d) occupied", i, j, k);
                e->voxel[bp_vox(i, j, k)] = (uint8_t)id;
            }
    for (int j = y; j < y + dy; j++)
        for (int i = x; i < x + dx; i++) {
            BP_CHECK(e->height[bp_cell(i, j)] < z + dz, "I-S6 column (%d,%d) height would not increase", i, j);
            e->height[bp_cell(i, j)] = (uint8_t)(z + dz);
        }

    BpPlaced *p = &e->placed[e->n_placed];
    p->inst = (uint8_t)e->cursor;
    p->o = (uint8_t)o;
    p->dx = (uint8_t)dx; p->dy = (uint8_t)dy; p->dz = (uint8_t)dz;
    p->x = (uint8_t)x; p->y = (uint8_t)y; p->z = (uint8_t)z;
    e->n_placed++;
    e->vol_placed += dx * dy * dz;
}

#ifdef DEBUG

static void bp_check_state(const Env *e) {
    BP_CHECK(e->n_placed <= e->n_boxes && e->n_boxes <= BP_MAX_BOXES, "I-S5 counts");
    for (int i = 0; i < e->n_placed; i++)
        BP_CHECK(e->placed[i].inst == i && e->placed[i].inst < e->n_boxes, "I-S5 placed %d -> instance %d", i, e->placed[i].inst);
    int occ = 0;
    for (int v = 0; v < BP_VOLUME; v++) {
        occ += (e->voxel[v] != 0);
        BP_CHECK(e->voxel[v] <= e->n_placed, "I-S5 owner %d > n_placed", e->voxel[v]);
    }
    BP_CHECK(occ == e->vol_placed, "I-S2 voxels %d vs volume %d", occ, e->vol_placed);
    BP_CHECK(e->vol_placed <= BP_VOLUME, "I-S6 utilization > 1");
    for (int y = 0; y < BP_SIZE; y++)
        for (int x = 0; x < BP_SIZE; x++) {
            int h = e->height[bp_cell(x, y)], top = 0;
            BP_CHECK(h <= BP_SIZE, "I-S1 height %d", h);
            for (int z = 0; z < BP_SIZE; z++) if (e->voxel[bp_vox(x, y, z)]) top = z + 1;
            BP_CHECK(h == top, "I-S4 column (%d,%d) height %d voxel top %d", x, y, h, top);
        }
}

static void bp_check_mask(const Env *e) {
    for (int a = 0; a < BP_ACT_TOTAL; a++)
        BP_CHECK(e->agents[0].action_mask[a] == (unsigned char)bp_is_legal(e, a), "I-M3 mask[%d]", a);
}
#endif

static int bp_policy_random(Env *e) {
    const unsigned char *m = e->agents[0].action_mask;
    int pop = 0;
    for (int a = 0; a < BP_ACT_TOTAL; a++) pop += m[a];
    int k = bp_below(&e->pol_rng, pop);
    for (int a = 0; a < BP_ACT_TOTAL; a++) if (m[a] && k-- == 0) return a;
    return -1;
}

static inline long bp_dbl_key(const Env *e, int a) {
    int o, x, y;
    bp_decode(a, &o, &x, &y);
    return (((long)e->rest_z[a] * BP_SIZE + y) * BP_SIZE + x) * BP_NORIENT + o;
}

static int bp_policy_dbl(Env *e) {
    const unsigned char *m = e->agents[0].action_mask;
    int best = -1; long bk = 0;
    for (int a = 0; a < BP_ACT_TOTAL; a++) {
        if (!m[a]) continue;
        long k = bp_dbl_key(e, a);
        if (best < 0 || k < bk) { best = a; bk = k; }
    }
    return best;
}

static inline int bp_h_after(const Env *e, int i, int j, int x, int y, int dx, int dy, int top) {
    if (i >= x && i < x + dx && j >= y && j < y + dy) return top;
    return e->height[bp_cell(i, j)];
}

static int bp_bumpiness_after(const Env *e, int x, int y, int dx, int dy, int top) {
    int b = 0;
    for (int j = 0; j < BP_SIZE; j++)
        for (int i = 0; i < BP_SIZE; i++) {
            int h = bp_h_after(e, i, j, x, y, dx, dy, top);
            if (i + 1 < BP_SIZE) b += abs(h - bp_h_after(e, i + 1, j, x, y, dx, dy, top));
            if (j + 1 < BP_SIZE) b += abs(h - bp_h_after(e, i, j + 1, x, y, dx, dy, top));
        }
    return b;
}

static int bp_policy_flat(Env *e) {
    const unsigned char *m = e->agents[0].action_mask;
    const BpBox *b = &e->instance[e->cursor];
    int best = -1, bb = 0; long bk = 0;
    for (int a = 0; a < BP_ACT_TOTAL; a++) {
        if (!m[a]) continue;
        int o, x, y, dx, dy, dz;
        bp_decode(a, &o, &x, &y);
        bp_orient(b, o, &dx, &dy, &dz);
        int bump = bp_bumpiness_after(e, x, y, dx, dy, e->rest_z[a] + dz);
        long k = bp_dbl_key(e, a);
        if (best < 0 || bump < bb || (bump == bb && k < bk)) { best = a; bb = bump; bk = k; }
    }
    return best;
}

static void bp_compute_obs(Env *e) {
    obs_t *ob = e->agents[0].observations;
    const float inv = 1.0f / (float)BP_SIZE;
    for (int c = 0; c < BP_CELLS; c++) ob[c] = (float)e->height[c] * inv;
    for (int k = 0; k <= BP_PREVIEW_K; k++) {
        int i = e->cursor + k;
        obs_t *d = ob + BP_CELLS + 3 * k;
        if (i < e->n_boxes) {
            d[0] = (float)e->instance[i].d[0] * inv;
            d[1] = (float)e->instance[i].d[1] * inv;
            d[2] = (float)e->instance[i].d[2] * inv;
        } else {
            d[0] = d[1] = d[2] = 0.0f;
        }
    }
}

static void bp_accumulate(Env *e, const long causes[5]) {
    e->legal_sum += causes[BP_LEGAL];
    for (int k = 0; k < 5; k++) e->cause_sum[k] += causes[k];
}

static void bp_reset_seeded(Env *e, unsigned int seed) {
    memset(e->height, 0, sizeof(e->height));
    memset(e->voxel, 0, sizeof(e->voxel));
    e->n_placed = 0;
    e->cursor = 0;
    e->vol_placed = 0;
    e->legal_sum = 0;
    for (int k = 0; k < 5; k++) e->cause_sum[k] = 0;
    e->illegal = 0;
    e->ep_return = 0.0f;

    bp_generate(e, seed);

    long causes[5];
    int pop = bp_build_mask(e, causes);
    BP_CHECK(pop > 0, "I-M1 empty mask on an empty container");
    (void)pop;
    bp_accumulate(e, causes);
    bp_compute_obs(e);
#ifdef DEBUG
    bp_check_mask(e);
#endif
}

static void bp_add_log(Env *e) {
    int len = e->n_placed;
    float util = (float)e->vol_placed / (float)BP_VOLUME;
    memcpy(e->last_placed, e->placed, sizeof(BpPlaced) * (size_t)e->n_placed);
    e->last_n = e->n_placed;
    e->last_boxes = e->n_boxes;
    e->last_util = util;
    e->render_hold = 1;
    BP_CHECK(len >= 1, "zero-length episode");
    BP_CHECK(e->ep_return - util < 1e-4f && util - e->ep_return < 1e-4f,
             "I-L1 return %f vs util %f", (double)e->ep_return, (double)util);
    float denom = (float)len * (float)BP_ACT_TOTAL;
    e->log.perf                 += util;
    e->log.score                += util;
    e->log.utilization          += util;
    e->log.boxes_placed         += (float)e->n_placed;
    e->log.boxes_remaining      += (float)(e->n_boxes - e->n_placed);
    e->log.mean_legal_actions   += (float)e->legal_sum / (float)len;
    e->log.mask_cause_orient    += (float)e->cause_sum[BP_FAIL_ORIENT] / denom;
    e->log.mask_cause_footprint += (float)e->cause_sum[BP_FAIL_FOOTPRINT] / denom;
    e->log.mask_cause_height    += (float)e->cause_sum[BP_FAIL_HEIGHT] / denom;
    e->log.mask_cause_support   += (float)e->cause_sum[BP_FAIL_SUPPORT] / denom;
    e->log.illegal_actions      += (float)e->illegal / (float)len;
    e->log.n                    += 1.0f;
}

void puf_reset(Env *e) {
    unsigned int seed = e->eval_seeds > 0 ? (unsigned int)(e->eval_seeds + e->ep_count)
                                          : bp_xorshift(&e->rng);
    e->ep_count++;
    bp_reset_seeded(e, seed);
}

void puf_step(Env *e) {
    e->agents[0].rewards[0] = 0.0f;
    e->agents[0].terminals[0] = 0.0f;

    int a;
    switch (e->policy) {
        case BP_POLICY_RANDOM: a = bp_policy_random(e); break;
        case BP_POLICY_DBL:    a = bp_policy_dbl(e);    break;
        case BP_POLICY_FLAT:   a = bp_policy_flat(e);   break;
        default:               a = (int)e->agents[0].actions[0]; break;
    }

    const unsigned char *mask = e->agents[0].action_mask;
    if (a < 0 || a >= BP_ACT_TOTAL || !mask[a]) {
        e->illegal++;
        a = 0;
        while (!mask[a]) a++;
    }

    int vol_before = e->vol_placed;
    bp_place(e, a);
    float r = (float)(e->vol_placed - vol_before) / (float)BP_VOLUME;
    e->agents[0].rewards[0] = r;
    e->ep_return += r;
    e->cursor++;
#ifdef DEBUG
    bp_check_state(e);
#endif

    if (e->cursor == e->n_boxes) {
        BP_CHECK(e->vol_placed == BP_VOLUME, "exhausted below full volume");
        e->agents[0].terminals[0] = 1.0f;
        bp_add_log(e);
        puf_reset(e);
        return;
    }

    long causes[5];
    int pop = bp_build_mask(e, causes);
    if (pop == 0) {
        e->agents[0].terminals[0] = 1.0f;
        bp_add_log(e);
        puf_reset(e);
        return;
    }
    bp_accumulate(e, causes);
    bp_compute_obs(e);
#ifdef DEBUG
    bp_check_mask(e);
#endif
}

static inline void bp_print_layers(const Env *e) {
    for (int z = 0; z < BP_SIZE; z++) {
        int any = 0;
        for (int c = 0; c < BP_CELLS; c++) any |= e->voxel[z * BP_CELLS + c];
        if (!any) break;
        printf("z=%d\n", z);
        for (int y = BP_SIZE - 1; y >= 0; y--) {
            for (int x = 0; x < BP_SIZE; x++) {
                int v = e->voxel[bp_vox(x, y, z)];
                if (v) printf("%3d", v); else printf("  .");
            }
            printf("\n");
        }
    }
}

static inline void bp_print_mask(const Env *e) {
    const BpBox *b = &e->instance[e->cursor];
    printf("box %d/%d dims %dx%dx%d\n", e->cursor, e->n_boxes, b->d[0], b->d[1], b->d[2]);
    for (int o = 0; o < e->num_orient; o++) {
        int dx, dy, dz;
        bp_orient(b, o, &dx, &dy, &dz);
        printf("o=%d (%dx%dx%d)\n", o, dx, dy, dz);
        for (int y = BP_SIZE - 1; y >= 0; y--) {
            for (int x = 0; x < BP_SIZE; x++)
                printf("%c", e->agents[0].action_mask[bp_encode(o, x, y)] ? '#' : '.');
            printf("\n");
        }
    }
}

typedef struct {
    float yaw, pitch, dist;
    int paused, init;
} BpView;

static BpView bp_view;

static void bp_view_reset(void) {
    bp_view.yaw = 0.785f;
    bp_view.pitch = 0.55f;
    bp_view.dist = 24.0f;
}

static void bp_view_input(void) {
    if (!bp_view.init) { bp_view_reset(); bp_view.init = 1; }
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        bp_view.yaw -= d.x * 0.008f;
        bp_view.pitch += d.y * 0.008f;
        if (bp_view.pitch > 1.5f) bp_view.pitch = 1.5f;
        if (bp_view.pitch < -0.2f) bp_view.pitch = -0.2f;
    }
    bp_view.dist -= GetMouseWheelMove() * 1.5f;
    if (bp_view.dist < 8.0f) bp_view.dist = 8.0f;
    if (bp_view.dist > 60.0f) bp_view.dist = 60.0f;
    if (IsKeyPressed(KEY_SPACE)) bp_view.paused = !bp_view.paused;
    if (IsKeyPressed(KEY_R)) bp_view_reset();
    if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) exit(0);
}

static void bp_draw_scene(const BpPlaced *pl, int n, const char *hud, const char *hud2) {
    const float h = (float)BP_SIZE / 2.0f;
    Camera3D cam;
    cam.target = (Vector3){0.0f, 3.5f, 0.0f};
    cam.position = (Vector3){
        cam.target.x + bp_view.dist * cosf(bp_view.pitch) * sinf(bp_view.yaw),
        cam.target.y + bp_view.dist * sinf(bp_view.pitch),
        cam.target.z + bp_view.dist * cosf(bp_view.pitch) * cosf(bp_view.yaw)};
    cam.up = (Vector3){0.0f, 1.0f, 0.0f};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(cam);
    DrawGrid(BP_SIZE, 1.0f);
    DrawCubeWires((Vector3){0.0f, h, 0.0f}, (float)BP_SIZE, (float)BP_SIZE, (float)BP_SIZE,
                  (Color){120, 140, 140, 255});
    for (int i = 0; i < n; i++) {
        const BpPlaced *p = &pl[i];
        Vector3 c = {(float)p->x + (float)p->dx / 2.0f - h,
                     (float)p->z + (float)p->dz / 2.0f,
                     (float)p->y + (float)p->dy / 2.0f - h};
        Color col = ColorFromHSV((float)((i * 47) % 360), 0.45f, 0.95f);
        if (i == n - 1) col = ColorFromHSV((float)((i * 47) % 360), 0.75f, 1.0f);
        float s = 0.98f;
        DrawCube(c, (float)p->dx * s, (float)p->dz * s, (float)p->dy * s, col);
        DrawCubeWires(c, (float)p->dx * s, (float)p->dz * s, (float)p->dy * s, (Color){20, 30, 30, 255});
    }
    EndMode3D();
    DrawText(hud, 12, 12, 20, (Color){241, 241, 241, 255});
    if (hud2) DrawText(hud2, 12, 38, 20, (Color){255, 200, 90, 255});
    DrawText(bp_view.paused ? "PAUSED  space resume | drag rotate | wheel zoom | R reset"
                            : "space pause | drag rotate | wheel zoom | R reset",
             12, GetScreenHeight() - 28, 18, (Color){150, 170, 170, 255});
    EndDrawing();
    puf_web_vsync();
}

static void bp_show(const BpPlaced *pl, int n, const char *hud, const char *hud2, int frames) {
    for (int f = 0; f < frames || bp_view.paused; f++) {
        bp_view_input();
        bp_draw_scene(pl, n, hud, hud2);
    }
}

void puf_render(Env *e) {
    if (!IsWindowReady()) {
        InitWindow(900, 760, "BinPack");
        SetTargetFPS(60);
    }
    char buf[128];
    if (e->render_hold) {
        e->render_hold = 0;
        snprintf(buf, sizeof(buf), "episode done  util %.3f  placed %d/%d",
                 (double)e->last_util, e->last_n, e->last_boxes);
        bp_show(e->last_placed, e->last_n, buf, "full or no legal placement", 150);
        return;
    }
    const BpBox *b = &e->instance[e->cursor];
    snprintf(buf, sizeof(buf), "util %.3f  box %d/%d  next %dx%dx%d",
             (double)e->vol_placed / BP_VOLUME, e->cursor, e->n_boxes, b->d[0], b->d[1], b->d[2]);
    bp_show(e->placed, e->n_placed, buf, NULL, 7);
}

void puf_init(Env *e, Dict *kwargs) {
    e->num_agents = 1;
    int no = (int)dict_get(kwargs, "num_orientations");
    e->num_orient = (no == 6) ? 6 : 2;
    e->stability = (int)dict_get(kwargs, "stability") != 0;
    e->policy = (int)dict_get(kwargs, "policy");
    if (e->policy < BP_POLICY_NONE || e->policy > BP_POLICY_FLAT) e->policy = BP_POLICY_NONE;
    e->eval_seeds = (int)dict_get(kwargs, "eval_seeds");
    if (e->eval_seeds < 0) e->eval_seeds = 0;
    e->ep_count = 0;

    unsigned int base = e->rng;
    e->rng = bp_splitmix32(base);
    e->pol_rng = bp_splitmix32(base ^ 0x5bd1e995u);
    e->agents[0].policy = 0;
    memset(&e->log, 0, sizeof(Log));
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf",                 log->perf);
    dict_set(out, "score",                log->score);
    dict_set(out, "utilization",          log->utilization);
    dict_set(out, "boxes_placed",         log->boxes_placed);
    dict_set(out, "boxes_remaining",      log->boxes_remaining);
    dict_set(out, "mean_legal_actions",   log->mean_legal_actions);
    dict_set(out, "mask_cause_orient",    log->mask_cause_orient);
    dict_set(out, "mask_cause_footprint", log->mask_cause_footprint);
    dict_set(out, "mask_cause_height",    log->mask_cause_height);
    dict_set(out, "mask_cause_support",   log->mask_cause_support);
    dict_set(out, "illegal_actions",      log->illegal_actions);
    dict_set(out, "n",                    log->n);
}

void puf_close(Env *e) {
    (void)e;
}
