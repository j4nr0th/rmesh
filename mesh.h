//
// Created by jan on 15.6.2024.
//

#ifndef MESH_MESH_H
#define MESH_MESH_H
#include "geometry.h"
#include "err.h"


typedef unsigned point_id;

typedef struct curve_struct curve;
struct curve_struct
{
    point_id pt1;
    point_id pt2;
};

typedef struct surface_struct surface;
struct surface_struct
{
    curve cs;
    curve ce;
    curve cn;
    curve cw;
};

typedef struct mesh_struct mesh;
struct mesh_struct
{
    unsigned n_points;
    unsigned n_curves;
    unsigned n_surfaces;
    double* p_x;
    double* p_y;
    curve* p_curves;
    surface* p_surfaces;
};


error_id mesh_create(unsigned n_blocks, const mesh_block* blocks, mesh** p_out);

void mesh_destroy(mesh* mesh);



#endif //MESH_MESH_H
