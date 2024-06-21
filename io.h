#ifndef MESH_IO_H
#define MESH_IO_H

#include "mesh.h"

int save_nodes_to_file(const char* fname, unsigned n, const double* x, const double* y);

int save_lines_to_file(const char* fname, unsigned n, const curve* lines);

#endif //MESH_IO_H
