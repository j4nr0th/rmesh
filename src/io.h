#ifndef MESH_IO_H
#define MESH_IO_H

#include "mesh2d.h"

int save_nodes_to_file(const char *fname, unsigned n, const double *x, const double *y);

int save_lines_to_file(const char *fname, unsigned n, const line *lines);

void mesh2d_save_args(const char *fname, unsigned n_blocks, const mesh2d_block *blocks, const solver_config *cfg);

int mesh2d_load_args(const char *fname, unsigned *pn_blocks, mesh2d_block **pp_blocks, solver_config *cfg);

#endif // MESH_IO_H
