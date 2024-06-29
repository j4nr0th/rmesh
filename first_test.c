#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "io.h"
#include "mesh2d.h"

static void* wrap_alloc(void* state, size_t sz)
{
    (void)state;
    return malloc(sz);
}
static void* wrap_realloc(void* state, void* ptr, size_t sz)
{
    (void)state;
    return realloc(ptr, sz);
}
static void wrap_free(void* state, void* ptr)
{
    (void)state;
    free(ptr);
}

int main(void)
{
    printf("Hello, World!\n");

    enum {NB1 = 3, NB2 = 3};

    double fwd1[NB1];
    double fwd2[NB2];
    double fwd21[NB1];
    double bwd1[NB1];
    double bwd21[NB1];
    double bwd2[NB2];
    double zero1[NB1] = {0};
    double zero2[NB2] = {0};
    double one1[NB1] = {0};
    double one2[NB2] = {0};
    // double two1[NB1] = {0};
    double two2[NB2] = {0};

    for (unsigned i = 0; i < NB1; ++i)
    {
        double x = (double)i/(double)(NB1-1);
        fwd1[i] = x;
        bwd1[NB1-i-1] = x;
        fwd21[i] = x + 1;
        bwd21[NB1-i-1] = x + 1;
        one1[i] = 1;
        // two1[i] = 2;
    }
    for (unsigned i = 0; i < NB2; ++i)
    {
        double x = (double)i/(double)(NB2-1);
        fwd2[i] = x;
        bwd2[NB2-i-1] = x;
        one2[i] = 1;
        two2[i] = 2;
    }

    mesh2d_block mb1 = 
    {
        .label = "first",
        .bsouth = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB2, .x = fwd2, .y=zero2}},
        .bnorth = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB2, .x = bwd2, .y=one2}},
        .beast = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = one1, .y=fwd1}},
        .bwest = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = zero1, .y=bwd1}},
    };

    mesh2d m = {0};
    allocator a = {.alloc = wrap_alloc, .realloc = wrap_realloc, .free = wrap_free};
    double rx, ry;
    solver_config cfg =
        {
        .direct = 0,
        .tol = 1e-6,
        .smoother_rounds = 0,
        .max_iterations = 1000,
        .max_rounds = 10,
        .verbose = 1
        };
    error_id err = mesh2d_create_elliptical(1, &mb1, &cfg, &a, &m, &rx, &ry);
    assert(err == MESH_SUCCESS);
    printf("Error code was %u\n", err);
    (void)err;
    save_nodes_to_file("out.dat", m.n_points, m.p_x, m.p_y);
    mesh_destroy(&m, &a);

    mesh2d_block blocks[2];
    blocks[0] = (mesh2d_block)
    {
        .label = "first",
        .bsouth = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB2, .x = fwd2, .y=zero2}},
        // .bnorth = {.type = BOUNDARY_TYPE_CURVE,
        //     .curve = {.n = NB2, .x = bwd2, .y=one2}},
        .bnorth = {.type = BOUNDARY_TYPE_BLOCK,
            .block = {.n = NB2, .owner = 0, .owner_id = BOUNDARY_ID_NORTH, .target = 1, .target_id = BOUNDARY_ID_SOUTH,}},
        .beast = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = one1, .y=fwd1}},
        .bwest = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = zero1, .y=bwd1}},
    };
    blocks[1] = (mesh2d_block)
    {
        .label = "third",
        // .bsouth = {.type = BOUNDARY_TYPE_CURVE,
        //     .curve = {.n = NB2, .x = fwd2, .y=zero2}},
        .bsouth = {.type = BOUNDARY_TYPE_BLOCK,
            .block = {.n = NB2, .owner = 1, .owner_id = BOUNDARY_ID_SOUTH, .target = 0, .target_id = BOUNDARY_ID_NORTH,}},
        .bnorth = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB2, .x = bwd2, .y=two2}},
        .beast = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = one1, .y=fwd21}},
        .bwest = {.type = BOUNDARY_TYPE_CURVE,
            .curve = {.n = NB1, .x = zero1, .y=bwd21}},
    };
    err = mesh2d_create_elliptical(2, blocks, &cfg, &a, &m, &rx, &ry);
    assert(err == MESH_SUCCESS);
    printf("Error code was %u\n", err);

    save_nodes_to_file("out1.dat", m.n_points, m.p_x, m.p_y);
    mesh_destroy(&m, &a);

    unsigned nb;
    mesh2d_block* pb;
    int v = mesh2d_load_args("../circular.bin", &nb, &pb, &cfg);
    assert(v == 0);
    (void)v;

    err = mesh2d_create_elliptical(nb, pb, &cfg, &a, &m, &rx, &ry);
    assert(err == MESH_SUCCESS);
    printf("Error code was %u\n", err);
    free(pb[0].bnorth.curve.x);
    free(pb[0].bnorth.curve.y);
    free(pb[0].bsouth.curve.x);
    free(pb[0].bsouth.curve.y);
    free(pb[0].beast.curve.x);
    free(pb[0].beast.curve.y);
    free(pb[0].bwest.curve.x);
    free(pb[0].bwest.curve.y);

    free(pb);
    mesh_destroy(&m, &a);

    return 0;
}
