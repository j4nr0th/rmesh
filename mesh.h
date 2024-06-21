//
// Created by jan on 15.6.2024.
//

#ifndef MESH_MESH_H
#define MESH_MESH_H
#include "geometry.h"
#include "err.h"


typedef unsigned geo_id;

typedef struct curve_struct curve;
struct curve_struct
{
    geo_id pt1;
    geo_id pt2;
};

typedef struct surface_struct surface;
struct surface_struct
{
    curve cs;
    curve ce;
    curve cn;
    curve cw;
};

typedef struct block_info_struct block_info;
struct block_info_struct
{
    unsigned n1, n2;        //  Number of points on the mesh
    geo_id* points;         //  Indices of points of the block (size is n1 * n2)
    geo_id* lines;          //  Indices of lines of the block (size is n1 * (n2-1) + n2 * (n1 - 1))
    geo_id* surfaces;       //  Indices of the surfaces of the block (size is (n1 - 1) * (n2 - 1))
};

typedef struct mesh_struct mesh;
struct mesh_struct
{
    unsigned n_blocks;
    unsigned n_points;
    unsigned n_curves;
    unsigned n_surfaces;
    block_info* block_info;
    double* p_x;
    double* p_y;
    curve* p_curves;
    surface* p_surfaces;
};


error_id mesh_create(unsigned n_blocks, mesh_block* blocks, mesh* p_out);

void mesh_destroy(mesh* mesh);



#endif //MESH_MESH_H
