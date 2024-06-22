#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "io.h"
#include "mesh2d.h"

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
    enum {NB1 = 8, NB2 = 7, NR = 6};

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
    error_id e = mesh2d_create_elliptical(5, blocks, &m);
    assert(e == MESH_SUCCESS);
    if (e != MESH_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }
    save_nodes_to_file("bigger_pts.dat", m.n_points, m.p_x, m.p_y);
    save_lines_to_file("bigger_lns.dat", m.n_lines, m.p_lines);
    mesh_destroy(&m);
    return 0;
}
