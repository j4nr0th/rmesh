//
// Created by jan on 7.7.2024.
//
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../src/mesh2d.h"
#include "../src/io.h"

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

static void linespace(double x0, double x1, unsigned n, double* pout)
{
    if (n == 1)
    {
        pout[0] = (x0+x1)/2;
    }
    for (unsigned i = 0; i < n; ++i)
    {
        const double s = (double)i / (double)(n - 1);
        pout[i] = x0 * (1-s) + x1 * s;
    }
}

int main(void)
{
    enum {NB1 = 10, NB2 = 10, NR = 2};

    double angle_right[NB1];
    double angle_left[NB1];
    double angle_top[NB2];
    double angle_bottom[NB2];

    linespace(M_PI_2, M_PI, NB2, angle_top);
    linespace(M_PI, M_PI + M_PI_2, NB1, angle_left);
    linespace(M_PI + M_PI_2, 2 * M_PI, NB2, angle_bottom);
    linespace(0, M_PI_2,  NB1, angle_right);

    double xl[NB1], yl[NB1];
    double xr[NB1], yr[NB1];
    double xt[NB1], yt[NB1];
    double xb[NB1], yb[NB1];

    for (unsigned i = 0; i < NB1; ++i)
    {
        xl[i] = 2 * cos(angle_left[i]);
        yl[i] = 2 * sin(angle_left[i]);

        xr[i] = 2 * cos(angle_right[i]);
        yr[i] = 2 * sin(angle_right[i]);
    }

    for (unsigned i = 0; i < NB2; ++i)
    {
        xt[i] = 2 * cos(angle_top[i]);
        yt[i] = 2 * sin(angle_top[i]);

        xb[i] = 2 * cos(angle_bottom[i]);
        yb[i] = 2 * sin(angle_bottom[i]);
    }

    mesh2d_block blocks[5] = {0};

    blocks[0] = (mesh2d_block)
    {
        .label =  "center",
        .bnorth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB2, .owner = 0, .owner_id = BOUNDARY_ID_NORTH,
                                                                  .target = 3, .target_id = BOUNDARY_ID_SOUTH}},
        .beast = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB1, .owner = 0, .owner_id = BOUNDARY_ID_EAST,
                                                                  .target = 2, .target_id = BOUNDARY_ID_WEST}},
        .bwest = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB1, .owner = 0, .owner_id = BOUNDARY_ID_WEST,
                                                                  .target = 1, .target_id = BOUNDARY_ID_EAST}},
        .bsouth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB2, .owner = 0, .owner_id = BOUNDARY_ID_SOUTH,
                                                                  .target = 4, .target_id = BOUNDARY_ID_NORTH}}
    };
    blocks[1] = (mesh2d_block)
    {
        .label =  "left",
        .bnorth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 3, .target_id = BOUNDARY_ID_WEST,
                                                                  .owner = 1, .owner_id = BOUNDARY_ID_NORTH}},
        .beast = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB1, .target = 0, .target_id = BOUNDARY_ID_WEST,
                                                                  .owner = 1, .owner_id = BOUNDARY_ID_EAST}},
        .bwest = {.type = BOUNDARY_TYPE_CURVE, .curve = {.n = NB1, .x=xl, .y=yl}},
        .bsouth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 4, .target_id = BOUNDARY_ID_WEST,
                                                                  .owner = 1, .owner_id = BOUNDARY_ID_SOUTH}},
    };
    blocks[2] = (mesh2d_block)
    {
        .label =  "right",
        .bnorth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 3, .target_id = BOUNDARY_ID_EAST,
                                                                  .owner = 2, .owner_id = BOUNDARY_ID_NORTH}},
        .beast = {.type=BOUNDARY_TYPE_CURVE, .curve = {.n = NB1, .x=xr, .y=yr}},
        .bwest = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB1, .target = 0, .target_id = BOUNDARY_ID_EAST,
                                                                  .owner = 2, .owner_id = BOUNDARY_ID_WEST}},
        .bsouth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 4, .target_id = BOUNDARY_ID_EAST,
                                                                  .owner = 2, .owner_id = BOUNDARY_ID_SOUTH}}
    };
    blocks[3] = (mesh2d_block)
    {
        .label =  "top",
        .bnorth = {.type = BOUNDARY_TYPE_CURVE, .curve = {.n = NB2, .x=xt, .y=yt}},
        .beast = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 2, .target_id = BOUNDARY_ID_NORTH,
                                                                  .owner = 3, .owner_id = BOUNDARY_ID_EAST}},
        .bwest = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 1, .target_id = BOUNDARY_ID_NORTH,
                                                                  .owner = 3, .owner_id = BOUNDARY_ID_WEST}},
        .bsouth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB2, .target = 0, .target_id = BOUNDARY_ID_NORTH,
                                                                  .owner = 3, .owner_id = BOUNDARY_ID_SOUTH}}
    };
    blocks[4] = (mesh2d_block)
    {
        .label =  "bottom",
        .bnorth = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NB2, .target = 0, .target_id = BOUNDARY_ID_SOUTH,
                                                                  .owner = 4, .owner_id = BOUNDARY_ID_NORTH}},
        .beast = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 2, .target_id = BOUNDARY_ID_SOUTH,
                                                                  .owner = 4, .owner_id = BOUNDARY_ID_EAST}},
        .bwest = {.type=BOUNDARY_TYPE_BLOCK, .block = {.n = NR, .target = 1, .target_id = BOUNDARY_ID_SOUTH,
                                                                  .owner = 4, .owner_id = BOUNDARY_ID_WEST}},
        .bsouth = {.type = BOUNDARY_TYPE_CURVE, .curve = {.n = NB2, .x=xb, .y=yb}}
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
    error_id e = mesh2d_create_elliptical(5, blocks, &cfg, &a, &m, &rx, &ry);
    printf("Error code was %u\n", e);

    assert(e == MESH_SUCCESS);
    if (e != MESH_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    enum {ORDER = 2};
    geo_id* adjacent = calloc((2 * ORDER + 2) * (2 * ORDER + 2), sizeof(*adjacent));
    assert(adjacent);
    memset(adjacent, ~0, (2 * ORDER + 2) * (2 * ORDER + 2) * sizeof(*adjacent));

    enum {TEST_SURFACE_ID = 1};

    e = surface_centered_element(&m, TEST_SURFACE_ID, ORDER, adjacent);
    assert(e == MESH_SUCCESS);
    if (e != MESH_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    printf("Following surfaces are adjacent:\n");
    for (unsigned i = 0; i < 2 * ORDER + 1; ++i)
    {
        for (unsigned j = 0; j < 2 * ORDER + 1; ++j)
        {
            printf("%3d%s", adjacent[i * (2 * ORDER + 1) + j],  j != 2 * ORDER ? "|" : "");
        }
        printf("\n");
        if (i != 2 * ORDER)
        {
            for (unsigned j = 0; j < 2 * ORDER + 1; ++j)
            {
                printf("---");
                if (j != 2 * ORDER)
                {
                    printf("+");
                }
            }
        }
        printf("\n");
    }


    memset(adjacent, ~0, (2 * ORDER + 2) * (2 * ORDER + 2) * sizeof(*adjacent));


    e = surface_centered_element_points(&m, TEST_SURFACE_ID, ORDER, adjacent);
    assert(e == MESH_SUCCESS);
    if (e != MESH_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    printf("Following points form the surface element:\n");
    for (unsigned i = 0; i < 2 * ORDER + 2; ++i)
    {
        for (unsigned j = 0; j < 2 * ORDER + 2; ++j)
        {
            printf("%3d ", adjacent[i * (2 * ORDER + 2) + j]);
        }
        printf("\n");
    }
    free(adjacent);

    mesh_destroy(&m, &a);
    return 0;
}
