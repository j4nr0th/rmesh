//
// Created by jan on 15.6.2024.
//

#ifndef MESH_MESH_H
#define MESH_MESH_H
#include "geometry.h"
#include "err.h"


typedef struct line_struct line;
struct line_struct
{
    geo_id pt1;
    geo_id pt2;
};

typedef struct surface_struct surface;
struct surface_struct
{
    geo_id lines;
    geo_id linee;
    geo_id linen;
    geo_id linew;
};

typedef struct block_info_struct block_info;
struct block_info_struct
{
    //  Number of points on the mesh
    unsigned n1, n2;
    //  Indices of points of the block (size is n1 * n2)
    geo_id* points;
    //  Indices of lines of the block (size is n1 * (n2-1) + n2 * (n1 - 1))
    geo_id* lines;
    //  Indices of the surfaces of the block (size is (n1 - 1) * (n2 - 1))
    geo_id* surfaces;
    //  Indices of the first and last points in the mesh block. There might be some shared with other blocks, but those
    //  are not included in here
    unsigned first_pt, last_pt;
    //  Indices of mesh blocks which border this block and where they border it. -1 means that there is no block bordering it
    struct
    {
        int west, east, north, south;
    } neighboring_block_idx;

    //  Indices of the first and last lines in the mesh block. There might be some shared with other blocks, but those
    //  are not included in here
    unsigned first_ln, last_ln;
};

typedef struct mesh_struct mesh2d;
struct mesh_struct
{
    unsigned n_blocks;
    block_info* block_info;
    unsigned n_points;
    double* p_x;
    double* p_y;
    unsigned n_lines;
    line* p_lines;
    unsigned n_surfaces;
    surface* p_surfaces;
};

error_id mesh2d_check_blocks(unsigned n_blocks, const mesh2d_block* blocks);

error_id mesh2d_create_elliptical(unsigned n_blocks, const mesh2d_block* blocks, mesh2d* p_out);

void mesh_destroy(mesh2d* mesh);



#endif //MESH_MESH_H
