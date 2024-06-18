#include <stdio.h>
#include "io.h"
#include "mesh.h"

int main(void)
{
    printf("Hello, World!\n");

    enum {NB1 = 10, NB2 = 5};

    double fwd1[NB1];
    double fwd2[NB2];
    double bwd1[NB1];
    double bwd2[NB2];
    double zero1[NB1] = {0};
    double zero2[NB2] = {0};
    double one1[NB1] = {0};
    double one2[NB2] = {0};

    for (unsigned i = 0; i < NB1; ++i)
    {
        double x = (double)i/(double)(NB1-1);
        fwd1[i] = x;
        bwd1[NB1-i-1] = x;
        one1[i] = 1;
    }
    for (unsigned i = 0; i < NB2; ++i)
    {
        double x = (double)i/(double)(NB2-1);
        fwd2[i] = x;
        bwd2[NB2-i-1] = x;
        one2[i] = 1;
    }

    mesh_block mb1 = 
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

    mesh m;
    error_id err = mesh_create(1, &mb1, &m);
    save_nodes_to_file("out.dat", m.n_points, m.p_x, m.p_y);

    return 0;
}
