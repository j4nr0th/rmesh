//
// Created by jan on 15.6.2024.
//

#ifndef MESH_MESH_H
#define MESH_MESH_H
#include <stddef.h>

#include "geometry.h"
#include "err.h"
#include "defines.h"

#ifndef _MSC_BUILD
    #define _RMSH_ARRAY_ATTRIB(x) x
#else
    #define _RMSH_ARRAY_ATTRIB(x)
#endif

typedef struct line_struct line;
struct line_struct
{
    geo_id pt1;
    geo_id pt2;
};

typedef struct surface_struct surface;
struct surface_struct
{
    union
    {
        struct
        {
            geo_id lines;
            geo_id linee;
            geo_id linen;
            geo_id linew;
        };
        geo_id line_array[4];
    };
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
    struct
    {
        geo_id index;
        boundary_id id;
    } boundaries[BOUNDARY_COUNT];

    //  Indices of the first and last lines in the mesh block. There might be some shared with other blocks, but those
    //  are not included in here
    unsigned first_ln, last_ln;
};

typedef struct mesh_struct mesh2d;
struct mesh_struct
{
    //  Block information
    unsigned n_blocks;
    block_info* block_info;
    //  Point informaiton (positions)
    unsigned n_points;
    double* p_x;
    double* p_y;
    //  Line information (points that make them)
    unsigned n_lines;
    line* p_lines;
    //  Surface information (lines that make them)
    unsigned n_surfaces;
    surface* p_surfaces;
};

typedef struct solver_config_struct solver_config;
struct solver_config_struct
{
    int direct;
    double tol;
    unsigned smoother_rounds;
    unsigned max_iterations;
    unsigned max_rounds;
    int verbose;
};

typedef struct allocator_struct allocator;
struct allocator_struct
{
    void* (*alloc)(void* state, size_t sz);
    void* (*realloc)(void* state, void* ptr, size_t newsz);
    void (*free)(void* state, void* ptr);
};

INTERNAL_MODULE_FUNCTION
error_id mesh2d_check_blocks(unsigned n_blocks, const mesh2d_block* blocks);

INTERNAL_MODULE_FUNCTION
error_id mesh2d_create_elliptical(unsigned n_blocks, mesh2d_block* blocks, const solver_config* cfg, allocator* allocator, mesh2d* p_out, double* rx, double* ry);

INTERNAL_MODULE_FUNCTION
void mesh_destroy(mesh2d* mesh, allocator* allocator);

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_lines_info(
    const mesh2d* mesh, unsigned block, boundary_id boundary,
    const geo_id** p_first, unsigned* p_count, int* p_stride);

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_points_info(
    const mesh2d* mesh, unsigned block, boundary_id boundary,
    const geo_id** p_first, unsigned* p_count, int* p_stride);

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_surface_info(
    const mesh2d* mesh, unsigned block, boundary_id boundary,
    const geo_id** p_first, unsigned* p_count, int* p_stride);

INTERNAL_MODULE_FUNCTION
unsigned point_boundary_index(const block_info* block, const boundary_id id, const unsigned idx);

INTERNAL_MODULE_FUNCTION
unsigned point_boundary_index_flipped(const block_info* block, const boundary_id id, const unsigned idx);

INTERNAL_MODULE_FUNCTION
unsigned line_boundary_index(const block_info* block, const boundary_id id, const unsigned idx);

INTERNAL_MODULE_FUNCTION
unsigned line_boundary_index_reverse(const block_info* block, const boundary_id id, const unsigned idx);

INTERNAL_MODULE_FUNCTION
unsigned surface_boundary_index(const block_info* block, const boundary_id id, const unsigned idx);

INTERNAL_MODULE_FUNCTION
error_id surface_centered_element(const mesh2d* mesh, geo_id surface_id, unsigned order, geo_id out[_RMSH_ARRAY_ATTRIB((2 * order + 1)*(2 * order + 1))]);

INTERNAL_MODULE_FUNCTION
error_id surface_centered_element_points(const mesh2d* mesh, geo_id surface_id, unsigned order, geo_id out[_RMSH_ARRAY_ATTRIB((2 * order + 2)*(2 * order + 2))]);

INTERNAL_MODULE_FUNCTION
error_id line_centered_element(const mesh2d* mesh, geo_id line_id, unsigned order, geo_id out[_RMSH_ARRAY_ATTRIB(2 * order + 1)]);

#endif //MESH_MESH_H
