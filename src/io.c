#include "io.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int save_nodes_to_file(const char *fname, unsigned n, const double *x, const double *y)
{
    FILE *fout = fopen(fname, "w");
    if (!fout)
    {
        return -1;
    }

    fprintf(fout, "# x y\n");

    for (unsigned i = 0; i < n; ++i)
    {
        fprintf(fout, "%.15e %.15e\n", x[i], y[i]);
    }

    fclose(fout);
    return 0;
}

int save_lines_to_file(const char *fname, unsigned n, const line *lines)
{
    FILE *fout = fopen(fname, "w");
    if (!fout)
    {
        return -1;
    }

    fprintf(fout, "# n1 n2\n");

    for (unsigned i = 0; i < n; ++i)
    {
        fprintf(fout, "%+d %+d\n", lines[i].pt1, lines[i].pt2);
    }

    fclose(fout);
    return 0;
}

static inline void serialize_boundary(FILE *fout, const boundary *b)
{
    if (b->type == BOUNDARY_TYPE_BLOCK)
    {
        fprintf(fout, "%u %u %u %u\n", b->block.owner, b->block.owner_id, b->block.target, b->block.target_id);
    }
    else
    {
        for (unsigned j = 0; j < b->n; ++j)
        {
            fprintf(fout, "(%g, %g) ", b->curve.x[j], b->curve.y[j]);
        }
    }
}

static inline void deserialize_boundary(FILE *fout, boundary *b)
{
    if (b->type == BOUNDARY_TYPE_BLOCK)
    {
        fscanf(fout, "%u %u %u %u\n", &b->block.owner, &b->block.owner_id, &b->block.target, &b->block.target_id);
    }
    else
    {
        double *x = calloc(b->n, sizeof *x);
        double *y = calloc(b->n, sizeof *x);
        for (unsigned j = 0; j < b->n; ++j)
        {
            fscanf(fout, "(%lg, %lg) ", x + j, y + j);
        }
        b->curve.x = x;
        b->curve.y = y;
    }
}

void mesh2d_save_args(const char *fname, unsigned n_blocks, const mesh2d_block *blocks, const solver_config *cfg)
{
    FILE *fout = fopen(fname, "w");
    if (!fout)
    {
        return;
    }
    fprintf(fout, "Block cnt: %u\n", n_blocks);
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const mesh2d_block *b = blocks + i;
        fprintf(fout, "Block:\n\tNorth(%u,%u) ->", b->bnorth.type, b->bnorth.n);
        serialize_boundary(fout, &b->bnorth);
        fprintf(fout, "\n\tWest(%u,%u) ->", b->bwest.type, b->bwest.n);
        serialize_boundary(fout, &b->bwest);
        fprintf(fout, "\n\tSouth(%u,%u) ->", b->bsouth.type, b->bsouth.n);
        serialize_boundary(fout, &b->bsouth);
        fprintf(fout, "\n\tEast(%u,%u) ->", b->beast.type, b->beast.n);
        serialize_boundary(fout, &b->beast);
    }
    fprintf(fout, "\ncfg:\n\tdirect:%d\n\tmax_iter:%u\n\tmax_rnds: %u\n\tsmoother_rnds: %u\n\ttol: %g\n\tverbose: %d\n",
            cfg->direct, cfg->max_iterations, cfg->max_rounds, cfg->smoother_rounds, cfg->tol, cfg->verbose);

    fclose(fout);
}

int mesh2d_load_args(const char *fname, unsigned *pn_blocks, mesh2d_block **pp_blocks, solver_config *cfg)
{
    FILE *fout = fopen(fname, "r");
    if (!fout)
    {
        return -1;
    }
    unsigned n_blocks;
    fscanf(fout, "Block cnt: %u\n", &n_blocks);
    mesh2d_block *p_blocks = calloc(n_blocks, sizeof *p_blocks);
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        mesh2d_block *b = p_blocks + i;
        b->label = 0;
        fscanf(fout, "Block:\n\tNorth(%u,%u) ->", &b->bnorth.type, &b->bnorth.n);
        deserialize_boundary(fout, &b->bnorth);
        fscanf(fout, "\n\tWest(%u,%u) ->", &b->bwest.type, &b->bwest.n);
        deserialize_boundary(fout, &b->bwest);
        fscanf(fout, "\n\tSouth(%u,%u) ->", &b->bsouth.type, &b->bsouth.n);
        deserialize_boundary(fout, &b->bsouth);
        fscanf(fout, "\n\tEast(%u,%u) ->", &b->beast.type, &b->beast.n);
        deserialize_boundary(fout, &b->beast);
    }
    fscanf(fout, "\ncfg:\n\tdirect:%d\n\tmax_iter:%u\n\tmax_rnds: %u\n\tsmoother_rnds: %u\n\ttol: %lg\n\tverbose: %d\n",
           &cfg->direct, &cfg->max_iterations, &cfg->max_rounds, &cfg->smoother_rounds, &cfg->tol, &cfg->verbose);
    *pn_blocks = n_blocks;
    *pp_blocks = p_blocks;
    fclose(fout);
    return 0;
}
