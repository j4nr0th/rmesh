//
// Created by jan on 15.6.2024.
//

#include "mesh2d.h"
#include <assert.h>
#include <jmtx/double/matrices/sparse_column_compressed.h>
#include <jmtx/double/matrices/sparse_row_compressed_safe.h>
#include <jmtx/double/solvers/bicgstab_iteration.h>
#include <stdio.h>
#include <time.h>

#include <jmtx/double/decompositions/band_lu_decomposition.h>
#include <jmtx/double/decompositions/incomplete_lu_decomposition.h>
#include <jmtx/double/matrices/band_row_major_safe.h>
#include <jmtx/double/solvers/lu_solving.h>

#include <jmtx/double/matrices/sparse_conversion.h>

static inline unsigned find_boundary_interior_node(const block_info *blk, const boundary_id t, const unsigned idx)
{
    switch (t)
    {
    case BOUNDARY_ID_SOUTH:
        return idx + blk->n2;
    case BOUNDARY_ID_NORTH:
        return blk->n2 * blk->n1 - idx - 1 - blk->n2;
    case BOUNDARY_ID_EAST:
        return idx * blk->n2 + (blk->n2 - 1) - 1;
    case BOUNDARY_ID_WEST:
        return blk->n2 * (blk->n1 - 1) - idx * blk->n2 + 1;
    }
    return 0;
}

static inline void sort_array5(const uint32_t in[5], uint32_t order[5])
{
    for (unsigned i = 0; i < 5; ++i)
    {
        const uint32_t v = in[i];
        uint32_t idx = 0;
        for (unsigned j = 0; j < 5; ++j)
        {
            idx += in[j] < v;
        }
        order[i] = idx;
    }
}

static inline jmtx_result interior_point_equation(jmtxd_matrix_crs *mat, const unsigned idx, const unsigned left,
                                                  const unsigned right, const unsigned top, const unsigned btm,
                                                  double *vx, double *vy, const int sort)
{
    uint32_t indices[5] = {btm, left, idx, right, top};
    double values[5] = {1, 1, 1, 1, 1};
    if (sort)
    {
        int needs_sorting = 0;
        for (unsigned n = 0; n < 4; ++n)
        {
            if (indices[n] > indices[n + 1])
            {
                needs_sorting = 1;
                break;
            }
        }
        if (needs_sorting)
        {
            uint32_t sort_order[5];
            const uint32_t in_indices[5] = {btm, left, idx, right, top};
            sort_array5(in_indices, sort_order);
            for (unsigned i = 0; i < 5; ++i)
            {
                indices[sort_order[i]] = in_indices[i];
            }
            values[sort_order[2]] = -4;
        }
        else
        {
            values[2] = -4;
        }
    }
    else
    {
        values[2] = -4;
    }
    vx[idx] = 0;
    vy[idx] = 0;
    return jmtxd_matrix_crs_build_row(mat, idx, 5, indices, values);
}

static inline jmtx_result boundary_point_condition(jmtxd_matrix_crs *mat, const unsigned idx, const double x,
                                                   const double y, double *vx, double *vy)
{
    vx[idx] = x;
    vy[idx] = y;
    const uint32_t index = idx;
    const double value = 1;
    return jmtxd_matrix_crs_build_row(mat, idx, 1, &index, &value);
}

static inline jmtx_result solve_the_system_of_equations(
    const unsigned npts, const jmtxd_matrix_crs *mat, double xrhs[_RMSH_ARRAY_ATTRIB(const restrict static npts)],
    double yrhs[_RMSH_ARRAY_ATTRIB(const restrict static npts)], double out_x[_RMSH_ARRAY_ATTRIB(const restrict npts)],
    double out_y[_RMSH_ARRAY_ATTRIB(const restrict npts)], allocator *allocator,
    const jmtx_allocator_callbacks *allocator_callbacks, const solver_config *cfg, double *rx, double *ry)
{
    jmtx_result res = JMTX_RESULT_SUCCESS;
    jmtx_result r1 = JMTX_RESULT_SUCCESS, r2 = JMTX_RESULT_SUCCESS;
    memset(out_x, 0, sizeof *out_x * npts);
    memset(out_y, 0, sizeof *out_y * npts);

    //  Is the problem small enough to solve for it directly?
    if (cfg->direct)
    {
        jmtxd_matrix_brm *banded = NULL, *l = NULL, *u = NULL;
        if (cfg->verbose)
            printf("Running the direct solver on an %u by %u problem\n", npts, npts);
        res = jmtxd_convert_crs_to_brm(mat, &banded, allocator_callbacks);
        if (res != JMTX_RESULT_SUCCESS)
        {
            return res;
        }
        res = jmtxd_decompose_lu_brm(banded, &l, &u, allocator_callbacks);
        jmtxd_matrix_brm_destroy(banded);
        if (res != JMTX_RESULT_SUCCESS)
        {
            return res;
        }
        jmtxd_solve_direct_lu_brm(l, u, xrhs, out_x);
        jmtxd_solve_direct_lu_brm(l, u, yrhs, out_y);
        jmtxd_matrix_brm_destroy(l);
        jmtxd_matrix_brm_destroy(u);
        *rx = -1;
        *ry = -1;
        return JMTX_RESULT_SUCCESS;
    }

    jmtxd_solver_arguments argsx = {
        .in_convergence_criterion = cfg->tol,
        .in_max_iterations = cfg->max_iterations,
        .out_last_error = 1.0, //  This is here in case we don't run GMRESR
    };
    jmtxd_solver_arguments args_smoother = {
        .in_convergence_criterion = 0,
        .in_max_iterations = cfg->smoother_rounds,
        .out_last_error = 1.0, //  This is here in case we don't run GMRESR
    };
    jmtxd_solver_arguments argsy = {
        .in_convergence_criterion = cfg->tol,
        .in_max_iterations = cfg->max_iterations,
        .out_last_error = 1.0, //  This is here in case we don't run GMRESR
    };

    double *const aux1 = allocator->alloc(allocator, npts * sizeof *aux1);
    double *const aux2 = allocator->alloc(allocator, npts * sizeof *aux2);
    double *const aux3 = allocator->alloc(allocator, npts * sizeof *aux3);
    double *const aux4 = allocator->alloc(allocator, npts * sizeof *aux4);
    double *const aux5 = allocator->alloc(allocator, npts * sizeof *aux5);
    double *const aux6 = allocator->alloc(allocator, npts * sizeof *aux6);
    double *const aux7 = allocator->alloc(allocator, npts * sizeof *aux7);
    double *const aux8 = allocator->alloc(allocator, npts * sizeof *aux8);
    if (!aux1 || !aux2 || !aux3 || !aux4 || !aux5 || !aux6 || !aux7 || !aux8)
    {
        res = JMTX_RESULT_BAD_ALLOC;
        goto end;
    }

    jmtxd_matrix_crs *l, *u;
    jmtxd_matrix_ccs *ccs_u;

    res = jmtxd_decompose_ilu_crs(mat, &l, &ccs_u, allocator_callbacks);
    if (res != JMTX_RESULT_SUCCESS)
    {
        goto end;
    }
    res = jmtxd_convert_ccs_to_crs(ccs_u, &u, allocator_callbacks);
    jmtxd_matrix_ccs_destroy(ccs_u);
    if (res != JMTX_RESULT_SUCCESS)
    {
        jmtxd_matrix_crs_destroy(l);
        goto end;
    }

    for (unsigned i = 0; i < cfg->max_rounds; ++i)
    {
        if (cfg->verbose)
            printf("Running PILUBICGSTAB on an %u by %u problem\n", npts, npts);
        r1 = jmtxd_solve_iterative_pilubicgstab_crs(mat, l, u, xrhs, out_x, aux1, aux2, aux3, aux4, aux5, aux6, aux7,
                                                    aux8, &argsx);
        if (cfg->verbose)
            printf("PILUBICGSTAB for the x equation finished in %u iterations with an error of %g\n",
                   argsx.out_last_iteration, argsx.out_last_error);
        if (argsx.out_last_error < argsx.in_convergence_criterion)
        {
            break;
        }
        if (cfg->smoother_rounds)
        {
            r1 = jmtxd_solve_iterative_ilu_crs_precomputed(mat, l, u, xrhs, out_x, aux1, &args_smoother);
        }
    }

    for (unsigned i = 0; i < cfg->max_rounds; ++i)
    {
        if (cfg->verbose)
            printf("Running PILUBICGSTAB on an %u by %u problem\n", npts, npts);
        r2 = jmtxd_solve_iterative_pilubicgstab_crs(mat, l, u, yrhs, out_y, aux1, aux2, aux3, aux4, aux5, aux6, aux7,
                                                    aux8, &argsy);
        if (cfg->verbose)
            printf("PILUBICGSTAB for the y equation finished in %u iterations with an error of %g\n",
                   argsy.out_last_iteration, argsy.out_last_error);
        if (argsy.out_last_error < argsy.in_convergence_criterion)
        {
            break;
        }
        if (cfg->smoother_rounds)
        {
            r1 = jmtxd_solve_iterative_ilu_crs_precomputed(mat, l, u, yrhs, out_y, aux1, &args_smoother);
        }
    }

    jmtxd_matrix_crs_destroy(l);
    jmtxd_matrix_crs_destroy(u);

end:
    allocator->free(allocator, aux8);
    allocator->free(allocator, aux7);
    allocator->free(allocator, aux6);
    allocator->free(allocator, aux5);
    allocator->free(allocator, aux4);
    allocator->free(allocator, aux3);
    allocator->free(allocator, aux2);
    allocator->free(allocator, aux1);
    *rx = argsx.out_last_error;
    *ry = argsy.out_last_error;

    if (r1 != JMTX_RESULT_SUCCESS)
    {
        return r1;
    }
    if (r2 != JMTX_RESULT_SUCCESS)
    {
        return r2;
    }

    return res;
}

static inline void free_block_info(block_info *info, allocator *allocator)
{
    allocator->free(allocator, info->points);
    info->points = NULL;
    allocator->free(allocator, info->lines);
    info->lines = NULL;
    allocator->free(allocator, info->surfaces);
    info->surfaces = NULL;
}

INTERNAL_MODULE_FUNCTION
unsigned point_boundary_index(const block_info *block, const boundary_id id, const unsigned idx)
{
    //  There is a flip that should be done
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        return idx; //  from 0 to block->n2
    case BOUNDARY_ID_NORTH:
        return block->n2 * block->n1 - (idx + 1); //  from 0 to block->n2
    case BOUNDARY_ID_WEST:
        return block->n2 * (block->n1 - 1 - idx); //  from 0 to block->n1
    case BOUNDARY_ID_EAST:
        return idx * block->n2 + block->n2 - 1; //  from 0 to block->n1
    }
    return 0;
}

INTERNAL_MODULE_FUNCTION
unsigned point_boundary_index_flipped(const block_info *block, const boundary_id id, const unsigned idx)
{
    //  There is a flip that should be done
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        return block->n2 - 1 - idx; //  from 0 to block->n2
    case BOUNDARY_ID_NORTH:
        return block->n2 * (block->n1 - 1) + idx; //  from 0 to block->n2
    case BOUNDARY_ID_WEST:
        return block->n2 * idx; //  from 0 to block->n1
    case BOUNDARY_ID_EAST:
        return block->n2 * (block->n1 - idx) - 1; //  from 0 to block->n1
    }
    return 0;
}

INTERNAL_MODULE_FUNCTION
unsigned line_boundary_index(const block_info *block, const boundary_id id, const unsigned idx)
{
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        return idx; //  From 0 to block->n2 - 1
    case BOUNDARY_ID_NORTH:
        return (block->n1 - 1) * (block->n2 - 1) + idx; //  From 0 to block->n2 - 1
    case BOUNDARY_ID_WEST:
        return block->n1 * (block->n2 - 1) + idx; //    From 0 to block->n1 - 1
    case BOUNDARY_ID_EAST:
        return block->n1 * (block->n2 - 1) + (block->n1 - 1) * (block->n2 - 1) + idx; //    From 0 to block->n1 - 1
    }
    return 0;
}

INTERNAL_MODULE_FUNCTION
unsigned line_boundary_index_reverse(const block_info *block, const boundary_id id, const unsigned idx)
{
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        return ((block->n2 - 2) - idx); //  From 0 to block->n2 - 1
    case BOUNDARY_ID_NORTH:
        return ((block->n1) * (block->n2 - 1) - (idx + 1)); //  From 0 to block->n2 - 1
    case BOUNDARY_ID_WEST:
        return (block->n1 * (block->n2 - 1) + (block->n1 - 2 - idx)); //    From 0 to block->n1 - 1
    case BOUNDARY_ID_EAST:
        return block->n1 * (block->n2 - 1) + (block->n1 - 1) * (block->n2 - 1) +
               (block->n1 - 2 - idx); //    From 0 to block->n1 - 1
    }
    return 0;
}

INTERNAL_MODULE_FUNCTION
unsigned surface_boundary_index(const block_info *block, const boundary_id id, const unsigned idx)
{
    //  There is a flip that should be done
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        assert((int)idx >= 0 && (int)idx < (int)(block->n2 - 1));
        return idx; //  from 0 to block->n2 - 1
    case BOUNDARY_ID_NORTH:
        assert((int)idx >= 0 && (int)idx < (int)(block->n2 - 1));
        return (block->n2 - 1) * (block->n1 - 1) - (idx + 1); //  from 0 to block->n2 - 1
    case BOUNDARY_ID_WEST:
        assert((int)idx >= 0 && (int)idx < (int)(block->n1 - 1));
        return (block->n2 - 1) * (block->n1 - 2 - idx); //  from 0 to block->n1 - 1
    case BOUNDARY_ID_EAST:
        assert((int)idx >= 0 && (int)idx < (int)(block->n1 - 1));
        return idx * (block->n2 - 1) + block->n2 - 2; //  from 0 to block->n1 - 1
    }
    return 0;
}

static inline geo_id find_adjacent_surface(const block_info *infos, unsigned *p_block_id, geo_id surface_id,
                                           boundary_id neighbour)
{
    const unsigned block_id = *p_block_id;
    const unsigned local_id = surface_id - infos[block_id].surfaces[0];
    const div_t coords_pack = div(local_id, infos[block_id].n2 - 1);
    const unsigned base_row = (unsigned)coords_pack.quot;
    const unsigned base_col = (unsigned)coords_pack.rem;

    switch (neighbour)
    {
    case BOUNDARY_ID_EAST:
        if (base_col == infos[block_id].n2 - 2)
        {
            if (infos[block_id].boundaries[BOUNDARY_ID_EAST - 1].id == ~0u)
            {
                return INVALID_SURFACE;
            }

            unsigned iother = infos[block_id].boundaries[BOUNDARY_ID_EAST - 1].index;
            *p_block_id = iother;
            unsigned idx = surface_boundary_index(infos + iother, infos[block_id].boundaries[BOUNDARY_ID_EAST - 1].id,
                                                  infos[block_id].n1 - 2 - base_row); //  + 1;
            return infos[iother].surfaces[idx];
        }
        return infos[block_id].surfaces[local_id + 1];
    case BOUNDARY_ID_WEST:
        if (base_col == 0)
        {
            if (infos[block_id].boundaries[BOUNDARY_ID_WEST - 1].id == ~0u)
            {
                return INVALID_SURFACE;
            }

            unsigned iother = infos[block_id].boundaries[BOUNDARY_ID_WEST - 1].index;
            *p_block_id = iother;
            unsigned idx = surface_boundary_index(infos + iother, infos[block_id].boundaries[BOUNDARY_ID_WEST - 1].id,
                                                  infos[block_id].n1 - 2 - base_row); // + 1;
            return infos[iother].surfaces[idx];
        }
        return infos[block_id].surfaces[local_id - 1];
    case BOUNDARY_ID_NORTH:
        if (base_row == infos[block_id].n1 - 2)
        {
            if (infos[block_id].boundaries[BOUNDARY_ID_NORTH - 1].id == ~0u)
            {
                return INVALID_SURFACE;
            }
            unsigned iother = infos[block_id].boundaries[BOUNDARY_ID_NORTH - 1].index;
            *p_block_id = iother;
            unsigned idx = surface_boundary_index(infos + iother, infos[block_id].boundaries[BOUNDARY_ID_NORTH - 1].id,
                                                  infos[block_id].n2 - 2 - base_col); //  + 1;
            return infos[iother].surfaces[idx];
        }
        return infos[block_id].surfaces[local_id + infos[block_id].n2 - 1];
    case BOUNDARY_ID_SOUTH:
        if (base_row == 0)
        {
            if (infos[block_id].boundaries[BOUNDARY_ID_SOUTH - 1].id == ~0u)
            {
                return INVALID_SURFACE;
            }

            unsigned iother = infos[block_id].boundaries[BOUNDARY_ID_SOUTH - 1].index;
            *p_block_id = iother;
            unsigned idx = surface_boundary_index(infos + iother, infos[block_id].boundaries[BOUNDARY_ID_SOUTH - 1].id,
                                                  infos[block_id].n2 - 2 - base_col); //  + 1;
            return infos[iother].surfaces[idx];
        }
        return infos[block_id].surfaces[local_id - (infos[block_id].n2 - 1)];
    }
    return INVALID_SURFACE;
}

error_id surface_centered_element(const mesh2d *mesh, geo_id surface_id, unsigned order,
                                  geo_id out[_RMSH_ARRAY_ATTRIB((2 * order + 1) * (2 * order + 1))])
{
    if (order == 0)
    {
        *out = surface_id;
        return MESH_SUCCESS;
    }
    const geo_id absolute_id = abs(surface_id);
    const block_info *bi = mesh->block_info;
    geo_id local_pos = -1;
    unsigned iblk;
    for (iblk = 0; iblk < mesh->n_blocks; ++iblk)
    {
        const block_info *info = bi + iblk;
        local_pos = absolute_id - info->surfaces[0];
        //  Surface can't be in the block
        if (local_pos >= 0 || local_pos < (int)((info->n1 - 1) * (info->n2 - 1)))
        {
            break;
        }
    }
    if (iblk == mesh->n_blocks)
    {
        return MESH_INDEX_OUT_OF_BOUNDS;
    }
    // const div_t coords_pack = div(local_pos, bi[iblk].n2);
    // const int base_row = (int)coords_pack.quot;
    // const int base_col = (int)coords_pack.rem;

    static const boundary_id move_directions[BOUNDARY_COUNT] = {BOUNDARY_ID_SOUTH, BOUNDARY_ID_EAST, BOUNDARY_ID_NORTH,
                                                                BOUNDARY_ID_WEST};
    const int move_offset[BOUNDARY_COUNT] = {-(2 * order + 1), +1, +(2 * order + 1), -1};
    static const boundary_id search_directions[BOUNDARY_COUNT] = {BOUNDARY_ID_EAST, BOUNDARY_ID_NORTH, BOUNDARY_ID_WEST,
                                                                  BOUNDARY_ID_SOUTH};
    const int search_offset[BOUNDARY_COUNT] = {
        +1,
        +(2 * order + 1),
        -1,
        -(2 * order + 1),
    };
    const unsigned center = (2 * order + 1) * order + order;
    //  Middle of array
    out[center] = absolute_id;
    for (unsigned i = 0; i < BOUNDARY_COUNT; ++i)
    {
        const boundary_id move = move_directions[i];
        const int mo = move_offset[i];
        const boundary_id search = search_directions[i];
        const int so = search_offset[i];
        unsigned block = iblk;
        geo_id surf = absolute_id;
        for (unsigned j = 0; j < order; ++j)
        {
            geo_id new_surf;
            if (surf != INVALID_SURFACE)
            {
                new_surf = find_adjacent_surface(bi, &block, surf, move);
                // printf("Block %d borders on %d on %s\n", surf, new_surf, boundary_id_to_str(move));
            }
            else
            {
                new_surf = surf;
            }
            out[center + (j + 1) * mo] = new_surf;
            surf = new_surf;
            unsigned new_block = block;
            for (unsigned k = 0; k < order; ++k)
            {
                geo_id new_new_surf;
                if (new_surf != INVALID_SURFACE)
                {
                    new_new_surf = find_adjacent_surface(bi, &new_block, new_surf, search);
                    // printf("Block %d borders on %d on %s\n", new_surf, new_new_surf, boundary_id_to_str(search));
                    // new_surf = new_new_surf;
                }
                else
                {
                    new_new_surf = new_surf;
                }
                out[center + (j + 1) * mo + (k + 1) * so] = new_new_surf;
                new_surf = new_new_surf;
            }
        }
    }

    return MESH_SUCCESS;
}

static inline geo_id surface_sw_point(const mesh2d *mesh, const geo_id surface)
{
    if (surface == INVALID_SURFACE)
    {
        return INVALID_POINT;
    }
    const unsigned surface_idx = abs(surface) - 1;
    const geo_id s_line = mesh->p_surfaces[surface_idx].lines;
    const unsigned line_idx = abs(s_line) - 1;
    geo_id end_pt;
    if (s_line * surface > 0) // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt1;
    }
    else // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt2;
    }
    return end_pt;
}
static inline geo_id surface_se_point(const mesh2d *mesh, const geo_id surface)
{
    if (surface == INVALID_SURFACE)
    {
        return INVALID_POINT;
    }
    const unsigned surface_idx = abs(surface) - 1;
    const geo_id s_line = mesh->p_surfaces[surface_idx].lines;
    const unsigned line_idx = abs(s_line) - 1;
    geo_id end_pt;
    if (s_line * surface > 0) // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt2;
    }
    else // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt1;
    }
    return end_pt;
}
static inline geo_id surface_nw_point(const mesh2d *mesh, const geo_id surface)
{
    if (surface == INVALID_SURFACE)
    {
        return INVALID_POINT;
    }
    const unsigned surface_idx = abs(surface) - 1;
    const geo_id n_line = mesh->p_surfaces[surface_idx].linen;
    const unsigned line_idx = abs(n_line) - 1;
    geo_id end_pt;
    if (n_line * surface > 0) // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt2;
    }
    else // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt1;
    }
    return end_pt;
}
static inline geo_id surface_ne_point(const mesh2d *mesh, const geo_id surface)
{
    if (surface == INVALID_SURFACE)
    {
        return INVALID_POINT;
    }
    const unsigned surface_idx = abs(surface) - 1;
    const geo_id n_line = mesh->p_surfaces[surface_idx].linen;
    const unsigned line_idx = abs(n_line) - 1;
    geo_id end_pt;
    if (n_line * surface > 0) // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt1;
    }
    else // Orientation not reversed
    {
        end_pt = mesh->p_lines[line_idx].pt2;
    }
    return end_pt;
}

error_id surface_centered_element_points(const mesh2d *mesh, geo_id surface_id, unsigned order,
                                         geo_id out[_RMSH_ARRAY_ATTRIB((2 * order + 2) * (2 * order + 2))])
{
    const error_id r = surface_centered_element(mesh, surface_id, order, out);
    if (r != MESH_SUCCESS)
    {
        return r;
    }
    unsigned len_in = (2 * order + 1) * (2 * order + 1);
    unsigned len_todo = (2 * order + 2) * (2 * order + 2);
    //  Write the last row
    for (unsigned i = 0; i < 2 * order + 1; ++i)
    {
        out[len_todo - 1] = surface_ne_point(mesh, out[len_in - 1 - i]);
        // len_in -= 1;
        len_todo -= 1;
        assert(len_todo >= len_in);
    }
    //  Don't forget the NW corner
    out[len_todo - 1] = surface_nw_point(mesh, out[len_in - 2 * order - 1]);
    len_todo -= 1;
    assert(len_todo >= len_in);

    //  Now do the actual rows
    for (unsigned row = 2 * order + 1; row > 0; --row)
    {
        //  First add the SE corner
        out[len_todo - 1] = surface_se_point(mesh, out[len_in - 1]);
        len_todo -= 1;
        assert(len_todo >= len_in);

        //  Now add all the others
        for (unsigned col = 2 * order + 1; col > 0; --col)
        {
            assert(len_todo >= len_in);
            out[len_todo - 1] = surface_sw_point(mesh, out[len_in - 1]);
            len_todo -= 1;
            len_in -= 1;
        }
    }
    assert(len_in == 0);
    assert(len_todo == 0);

    return MESH_SUCCESS;
}

static inline void deal_with_line_boundary(const boundary_block *boundary, const block_info *info_owner,
                                           const block_info *info_target)
{
    //  When this is 1 we reverse, when it's -1 we do not
    int reversed = boundary->owner_id == boundary->target_id;
    if (!reversed)
    {
        switch (boundary->owner_id)
        {
        case BOUNDARY_ID_NORTH:
            reversed = (boundary->target_id == BOUNDARY_ID_WEST);
            break;
        case BOUNDARY_ID_WEST:
            reversed = (boundary->target_id == BOUNDARY_ID_NORTH);
            break;
        case BOUNDARY_ID_SOUTH:
            reversed = (boundary->target_id == BOUNDARY_ID_EAST);
            break;
        case BOUNDARY_ID_EAST:
            reversed = (boundary->target_id == BOUNDARY_ID_SOUTH);
            break;
        default:
            break;
        }
    }

    if (reversed)
    {
        for (unsigned i = 0; i < boundary->n - 1; ++i)
        {
            const geo_id iother = -info_target->lines[line_boundary_index_reverse(info_target, boundary->target_id, i)];
            const unsigned this_idx = line_boundary_index(info_owner, boundary->owner_id, i);
            info_owner->lines[this_idx] = iother;
        }
    }
    else
    {
        for (unsigned i = 0; i < boundary->n - 1; ++i)
        {
            const geo_id iother = info_target->lines[line_boundary_index(info_target, boundary->target_id, i)];
            const unsigned this_idx = line_boundary_index(info_owner, boundary->owner_id, i);
            info_owner->lines[this_idx] = iother;
        }
    }
}

typedef struct mesh2d_geo_args_struct mesh_geo_args;
struct mesh2d_geo_args_struct
{
    size_t point_cnt;
    size_t max_lines;
    size_t max_surfaces;
    unsigned *block_offsets;
    const double *xnodal;
    const double *ynodal;
    block_info *info;
};

static inline void deal_with_point_boundary(const unsigned i, const boundary_block *boundary,
                                            const block_info *info_owner, const block_info *info_target, double *newx,
                                            double *newy, unsigned *division_factor, const mesh_geo_args *args)
{
    for (unsigned j = 0; j < boundary->n; ++j)
    {
        unsigned other_idx = info_target->points[point_boundary_index_flipped(info_target, (boundary->target_id), j)];
        unsigned this_idx = point_boundary_index(info_owner, boundary->owner_id, j);
        newx[other_idx] += args->xnodal[this_idx + args->block_offsets[i]];
        newy[other_idx] += args->ynodal[this_idx + args->block_offsets[i]];
        info_owner->points[this_idx] = other_idx;
        division_factor[other_idx] += 1;
    }
}

/**
 * Checks if the curve boundaries represent the same curve and should thus be merged
 * @param b1 first boundary
 * @param b2 second boundary
 * @return nonzero if the two boundary curves match and are indeed representing the same curve
 */
static inline int are_curve_boundaries_the_same(const boundary_curve *b1, const boundary_curve *b2)
{
    if (b1->n != b2->n)
    {
        return 0;
    }
    const unsigned n = b1->n;
    for (unsigned i = 0; i < n; ++i)
    {
        if (b1->x[i] != b2->x[n - 1 - i] || b1->y[i] != b2->y[n - 1 - i])
        {
            return 0;
        }
    }
    return 1;
}

static error_id generate_mesh2d_from_geometry(unsigned n_blocks, mesh2d_block *blocks, mesh2d *p_out,
                                              allocator *allocator, const mesh_geo_args args)
{
    //  Remove duplicate points by averaging over them
    block_info *info = args.info;
    unsigned unique_pts = 0;
    unsigned *division_factor = allocator->alloc(allocator, args.point_cnt * sizeof *division_factor);
#ifndef _NDEBUG
    memset(division_factor, 0, sizeof *division_factor * args.point_cnt);
#endif
    double *newx = allocator->alloc(allocator, args.point_cnt * sizeof *newx);
    double *newy = allocator->alloc(allocator, args.point_cnt * sizeof *newy);
    line *line_array = allocator->alloc(allocator, args.max_lines * sizeof(*line_array));
    unsigned surf_count = 0;
    surface *surfaces = allocator->alloc(allocator, args.max_surfaces * sizeof *surfaces);
    if (!division_factor || !newx || !newy || !line_array || !surfaces)
    {
        allocator->free(allocator, surfaces);
        allocator->free(allocator, line_array);
        allocator->free(allocator, newy);
        allocator->free(allocator, newx);
        allocator->free(allocator, division_factor);
        return MESH_ALLOCATION_FAILED;
    }
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        mesh2d_block *b = blocks + i;
        if (b->bnorth.type == BOUNDARY_TYPE_CURVE)
        {
            const boundary_curve *bc = &b->bnorth.curve;
            unsigned j;
            boundary_id bid = 0;
            for (j = 0; j < i; ++j)
            {
                const mesh2d_block *other = blocks + j;
                if (other->bnorth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bnorth.curve))
                {
                    bid = BOUNDARY_ID_NORTH;
                    break;
                }
                if (other->bsouth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bsouth.curve))
                {
                    bid = BOUNDARY_ID_SOUTH;
                    break;
                }
                if (other->beast.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->beast.curve))
                {
                    bid = BOUNDARY_ID_EAST;
                    break;
                }
                if (other->bwest.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->bwest.curve))
                {
                    bid = BOUNDARY_ID_WEST;
                    break;
                }
            }
            if (j != i)
            {
                // printf("Blocks %u and %u share boundaries %s and %s\n", i, j, boundary_id_to_str(BOUNDARY_ID_NORTH),
                // boundary_id_to_str(bid));
                assert(bid != 0);
                b->bnorth.type = BOUNDARY_TYPE_BLOCK;
                b->bnorth.block = (boundary_block){
                    .n = b->bnorth.n, .owner = i, .owner_id = BOUNDARY_ID_NORTH, .target = j, .target_id = bid};
            }
        }
        if (b->bsouth.type == BOUNDARY_TYPE_CURVE)
        {
            const boundary_curve *bc = &b->bsouth.curve;
            unsigned j;
            boundary_id bid = 0;
            for (j = 0; j < i; ++j)
            {
                const mesh2d_block *other = blocks + j;
                if (other->bnorth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bnorth.curve))
                {
                    bid = BOUNDARY_ID_NORTH;
                    break;
                }
                if (other->bsouth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bsouth.curve))
                {
                    bid = BOUNDARY_ID_SOUTH;
                    break;
                }
                if (other->beast.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->beast.curve))
                {
                    bid = BOUNDARY_ID_EAST;
                    break;
                }
                if (other->bwest.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->bwest.curve))
                {
                    bid = BOUNDARY_ID_WEST;
                    break;
                }
            }
            if (j != i)
            {
                // printf("Blocks %u and %u share boundaries %s and %s\n", i, j, boundary_id_to_str(BOUNDARY_ID_SOUTH),
                // boundary_id_to_str(bid));
                assert(bid != 0);
                b->bsouth.type = BOUNDARY_TYPE_BLOCK;
                b->bsouth.block = (boundary_block){
                    .n = b->bsouth.n, .owner = i, .owner_id = BOUNDARY_ID_SOUTH, .target = j, .target_id = bid};
            }
        }
        if (b->beast.type == BOUNDARY_TYPE_CURVE)
        {
            const boundary_curve *bc = &b->beast.curve;
            unsigned j;
            boundary_id bid = 0;
            for (j = 0; j < i; ++j)
            {
                const mesh2d_block *other = blocks + j;
                if (other->bnorth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bnorth.curve))
                {
                    bid = BOUNDARY_ID_NORTH;
                    break;
                }
                if (other->bsouth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bsouth.curve))
                {
                    bid = BOUNDARY_ID_SOUTH;
                    break;
                }
                if (other->beast.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->beast.curve))
                {
                    bid = BOUNDARY_ID_EAST;
                    break;
                }
                if (other->bwest.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->bwest.curve))
                {
                    bid = BOUNDARY_ID_WEST;
                    break;
                }
            }
            if (j != i)
            {
                assert(bid != 0);
                // printf("Blocks %u and %u share boundaries %s and %s\n", i, j, boundary_id_to_str(BOUNDARY_ID_EAST),
                // boundary_id_to_str(bid));
                b->beast.type = BOUNDARY_TYPE_BLOCK;
                b->beast.block = (boundary_block){
                    .n = b->beast.n, .owner = i, .owner_id = BOUNDARY_ID_EAST, .target = j, .target_id = bid};
            }
        }
        if (b->bwest.type == BOUNDARY_TYPE_CURVE)
        {
            const boundary_curve *bc = &b->bwest.curve;
            unsigned j;
            boundary_id bid = 0;
            for (j = 0; j < i; ++j)
            {
                const mesh2d_block *other = blocks + j;
                if (other->bnorth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bnorth.curve))
                {
                    bid = BOUNDARY_ID_NORTH;
                    break;
                }
                if (other->bsouth.type == BOUNDARY_TYPE_CURVE &&
                    are_curve_boundaries_the_same(bc, &other->bsouth.curve))
                {
                    bid = BOUNDARY_ID_SOUTH;
                    break;
                }
                if (other->beast.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->beast.curve))
                {
                    bid = BOUNDARY_ID_EAST;
                    break;
                }
                if (other->bwest.type == BOUNDARY_TYPE_CURVE && are_curve_boundaries_the_same(bc, &other->bwest.curve))
                {
                    bid = BOUNDARY_ID_WEST;
                    break;
                }
            }
            if (j != i)
            {
                // printf("Blocks %u and %u share boundaries %s and %s\n", i, j, boundary_id_to_str(BOUNDARY_ID_WEST),
                // boundary_id_to_str(bid));
                assert(bid != 0);
                b->bwest.type = BOUNDARY_TYPE_BLOCK;
                b->bwest.block = (boundary_block){
                    .n = b->bwest.n, .owner = i, .owner_id = BOUNDARY_ID_WEST, .target = j, .target_id = bid};
            }
        }
    }
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const mesh2d_block *b = blocks + i;
        block_info *bi = info + i;
        for (unsigned j = 0; j < BOUNDARY_COUNT; ++j)
        {
            const boundary *bnd = b->bnd_array + j;
            if (bnd->type == BOUNDARY_TYPE_BLOCK)
            {
                bi->boundaries[j].id = bnd->block.target_id;
                bi->boundaries[j].index = bnd->block.target;
            }
        }
    }

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info *bi = info + i;
        const mesh2d_block *b = blocks + i;
        unsigned iother;
        int has_n = 0, has_s = 0, has_e = 0, has_w = 0;
        bi->first_pt = unique_pts;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target)) < i)
        {
            deal_with_point_boundary(i, &b->bnorth.block, bi, info + iother, newx, newy, division_factor, &args);
            has_n = 1;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target)) < i)
        {
            deal_with_point_boundary(i, &b->bsouth.block, bi, info + iother, newx, newy, division_factor, &args);
            has_s = 1;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target)) < i)
        {
            deal_with_point_boundary(i, &b->beast.block, bi, info + iother, newx, newy, division_factor, &args);
            has_e = 1;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target)) < i)
        {
            deal_with_point_boundary(i, &b->bwest.block, bi, info + iother, newx, newy, division_factor, &args);
            has_w = 1;
        }
        // unsigned new_pts = bi->n1 * bi->n2 - duplicate;
        // unsigned offset = unique_pts;
        for (unsigned row = has_s; row < bi->n1 - has_n; ++row)
        {
            for (unsigned col = has_w; col < bi->n2 - has_e; ++col)
            {
                geo_id idx = col + row * bi->n2;
                assert(bi->points[idx] == ~0);
                bi->points[idx] = unique_pts;
                newx[unique_pts] = args.xnodal[args.block_offsets[i] + idx];
                newy[unique_pts] = args.ynodal[args.block_offsets[i] + idx];
                assert(division_factor[unique_pts] == 0);
                division_factor[unique_pts] = 1;
                unique_pts += 1;
            }
        }

        bi->last_pt = unique_pts;
    }

    for (unsigned i = 0; i < unique_pts; ++i)
    {
        assert(division_factor[i] != 0);
        const double d = 1.0 / (double)division_factor[i];
        newx[i] *= d;
        newy[i] *= d;
    }
    if (unique_pts != args.point_cnt)
    {
        double *tmp = allocator->realloc(allocator, newx, unique_pts * sizeof *newx);
        assert(tmp);
        newx = tmp;
    }
    if (unique_pts != args.point_cnt)
    {
        double *tmp = allocator->realloc(allocator, newy, unique_pts * sizeof *newy);
        assert(tmp);
        newy = tmp;
    }
    allocator->free(allocator, division_factor);

    // Create mesh line info
    unsigned line_count = 1;
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info *bi = info + i;
        const mesh2d_block *b = blocks + i;
        unsigned has_n = 0, has_s = 0, has_e = 0, has_w = 0;
        unsigned iother;
        bi->first_ln = line_count;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target)) < i)
        {
            deal_with_line_boundary(&b->bnorth.block, info + i, info + iother);
            has_n = 1;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target)) < i)
        {
            deal_with_line_boundary(&b->bsouth.block, info + i, info + iother);
            has_s = 1;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target)) < i)
        {
            deal_with_line_boundary(&b->beast.block, info + i, info + iother);
            has_e = 1;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target)) < i)
        {
            deal_with_line_boundary(&b->bwest.block, info + i, info + iother);
            has_w = 1;
        }

        for (unsigned row = has_s; row < bi->n1 - has_n; ++row)
        {
            for (unsigned col = 0; col < bi->n2 - 1; ++col)
            {
                unsigned idx = row * bi->n2 + col;
                unsigned n1 = bi->points[idx];
                unsigned n2 = bi->points[idx + 1];
                bi->lines[row * (bi->n2 - 1) + col] = line_count;
                line_array[line_count - 1] = (line){.pt1 = n1, .pt2 = n2};
                line_count += 1;
            }
        }

        for (unsigned col = has_w; col < bi->n2 - has_e; ++col)
        {
            for (unsigned row = 0; row < bi->n1 - 1; ++row)
            {
                unsigned idx = row * bi->n2 + col;
                unsigned n1 = bi->points[idx];
                unsigned n2 = bi->points[idx + bi->n2];
                bi->lines[(bi->n2 - 1) * bi->n1 + col * (bi->n1 - 1) + row] = line_count;
                line_array[line_count - 1] = (line){.pt1 = n1, .pt2 = n2};
                line_count += 1;
            }
        }
#ifndef _NDEBUG
        for (unsigned j = 0; j < (bi->n2 - 1) * bi->n1 + bi->n2 * (bi->n1 - 1); ++j)
        {
            assert(bi->lines[j] != 0);
        }
#endif
        bi->last_ln = line_count;
    }
    line_count -= 1;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info *bi = info + i;
        for (unsigned row = 0; row < bi->n1 - 1; ++row)
        {
            for (unsigned col = 0; col < bi->n2 - 1; ++col)
            {
                geo_id btm = (geo_id)(bi->lines[col + row * (bi->n2 - 1)]);
                geo_id top = (geo_id)(bi->lines[col + (row + 1) * (bi->n2 - 1)]);
                geo_id lft = (geo_id)(bi->lines[bi->n1 * (bi->n2 - 1) + row + col * (bi->n1 - 1)]);
                geo_id rgt = (geo_id)(bi->lines[bi->n1 * (bi->n2 - 1) + row + (col + 1) * (bi->n1 - 1)]);
                assert(surf_count < args.max_surfaces);
                bi->surfaces[col + row * (bi->n2 - 1)] = (geo_id)(surf_count + 1);
                surfaces[surf_count] = (surface){.lines = +btm, .linee = +rgt, .linen = -top, .linew = -lft};
                surf_count += 1;
            }
        }
    }
    assert(surf_count == args.max_surfaces);
    p_out->n_blocks = n_blocks;
    p_out->p_x = newx;
    p_out->p_y = newy;
    p_out->block_info = info;
    p_out->n_points = unique_pts;
    p_out->n_lines = line_count;
    p_out->p_lines = line_array;
    p_out->n_surfaces = args.max_surfaces;
    p_out->p_surfaces = surfaces;
    surfaces = NULL;
    line_array = NULL;
    info = NULL;

    return MESH_SUCCESS;
}

static inline error_id check_boundary_consistency(const mesh2d_block *blocks, const boundary_block *bnd,
                                                  const unsigned idx)
{
    const boundary *other = NULL;
    switch (bnd->target_id)
    {
    case BOUNDARY_ID_NORTH:
        other = &blocks[bnd->target].bnorth;
        break;
    case BOUNDARY_ID_SOUTH:
        other = &blocks[bnd->target].bsouth;
        break;
    case BOUNDARY_ID_EAST:
        other = &blocks[bnd->target].beast;
        break;
    case BOUNDARY_ID_WEST:
        other = &blocks[bnd->target].bwest;
        break;
    }
    if (other == NULL)
    {
        return MESH_INVALID_BOUNDARY_ID;
    }
    if (other->n != bnd->n)
    {
        return MESH_BOUNDARY_SIZE_MISMATCH;
    }
    if (other->type == BOUNDARY_TYPE_CURVE && (unsigned)bnd->target >= idx)
    {
        return MESH_BOUNDARY_UNSORTED;
    }
    return MESH_SUCCESS;
}

INTERNAL_MODULE_FUNCTION
error_id mesh2d_check_blocks(unsigned n_blocks, const mesh2d_block *blocks)
{
    error_id ret = MESH_SUCCESS;
    for (unsigned iblk = 0; iblk < n_blocks; ++iblk)
    {
        const mesh2d_block *b = blocks + iblk;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK &&
            (ret = check_boundary_consistency(blocks, &b->bnorth.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK &&
            (ret = check_boundary_consistency(blocks, &b->bsouth.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK &&
            (ret = check_boundary_consistency(blocks, &b->beast.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK &&
            (ret = check_boundary_consistency(blocks, &b->bwest.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
    }
    return ret;
}

INTERNAL_MODULE_FUNCTION
error_id mesh2d_create_elliptical(unsigned n_blocks, mesh2d_block *blocks, const solver_config *cfg,
                                  allocator *allocator, mesh2d *p_out, double *rx, double *ry)
{
    jmtx_allocator_callbacks allocator_callbacks = {
        .alloc = allocator->alloc,
        .free = allocator->free,
        .realloc = allocator->realloc,
        .state = (void *)0xB16B00B135,
    };
    error_id ret = MESH_SUCCESS;
    unsigned point_cnt = 0;
    unsigned max_lines = 0;
    unsigned max_surfaces = 0;
    unsigned *block_offsets = allocator->alloc(allocator, n_blocks * sizeof *block_offsets);
    block_info *info = allocator->alloc(allocator, n_blocks * sizeof *info);
    if (!block_offsets || !info)
    {
        allocator->free(allocator, block_offsets);
        allocator->free(allocator, info);
        return MESH_ALLOCATION_FAILED;
    }
    memset(block_offsets, 0, n_blocks * sizeof(*block_offsets));
    for (unsigned iblk = 0; iblk < n_blocks; ++iblk)
    {
        const mesh2d_block *const blk = blocks + iblk;
        //  Check mesh boundaries and compute each block's size
        unsigned nnorth = blk->bnorth.n;
        unsigned nsouth = blk->bsouth.n;
        if (nnorth != nsouth)
        {
            return MESH_BOUNDARY_SIZE_MISMATCH;
        }
        unsigned n_east = blk->beast.n;
        unsigned n_west = blk->bwest.n;
        if (n_east != n_west)
        {
            return MESH_BOUNDARY_SIZE_MISMATCH;
        }

        size_t npts = nnorth * n_east;
        size_t n1 = n_east;
        size_t n2 = nnorth;
        info[iblk].n1 = n1;
        info[iblk].n2 = n2;
        info[iblk].points = allocator->alloc(allocator, n1 * n2 * sizeof(*info[iblk].points));
        assert(info[iblk].points);
        memset(info[iblk].points, ~0, n1 * n2 * sizeof(*info[iblk].points));
        info[iblk].lines = allocator->alloc(allocator, (n1 * (n2 - 1) + (n1 - 1) * n2) * sizeof(*info[iblk].lines));
        assert(info[iblk].lines);
        info[iblk].surfaces = allocator->alloc(allocator, (n1 - 1) * (n2 - 1) * sizeof(*info[iblk].surfaces));
        assert(info[iblk].surfaces);
        if (iblk != n_blocks - 1)
        {
            block_offsets[iblk + 1] = npts + block_offsets[iblk];
        }
        for (unsigned i = 0; i < BOUNDARY_COUNT; ++i)
        {
            info[iblk].boundaries[i].id = (boundary_id)BOUNDARY_ID_INVALID;
            info[iblk].boundaries[i].index = INVALID_BLOCK;
        }
        point_cnt += npts;
        max_lines += (n1 - 1) * n2 + n1 * (n2 - 1);
        max_surfaces += (n1 - 1) * (n2 - 1);
    }

    double *xrhs = allocator->alloc(allocator, point_cnt * sizeof *xrhs);
    double *yrhs = allocator->alloc(allocator, point_cnt * sizeof *yrhs);
    double *xnodal = allocator->alloc(allocator, point_cnt * sizeof *xnodal);
    double *ynodal = allocator->alloc(allocator, point_cnt * sizeof *ynodal);
    if (!xrhs || !yrhs || !xnodal || !ynodal)
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            free_block_info(info + i, allocator);
        }
        allocator->free(allocator, xrhs);
        allocator->free(allocator, yrhs);
        allocator->free(allocator, xnodal);
        allocator->free(allocator, ynodal);
        allocator->free(allocator, block_offsets);
        allocator->free(allocator, info);
        return MESH_ALLOCATION_FAILED;
    }
    jmtxd_matrix_crs *system_matrix;
    jmtx_result res = jmtxds_matrix_crs_new(&system_matrix, point_cnt, point_cnt, 4 * point_cnt, &allocator_callbacks);
    if (res != JMTX_RESULT_SUCCESS)
    {
        allocator->free(allocator, ynodal);
        allocator->free(allocator, xnodal);
        allocator->free(allocator, yrhs);
        allocator->free(allocator, xrhs);
        allocator->free(allocator, block_offsets);
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            free_block_info(info + i, allocator);
        }
        allocator->free(allocator, info);
        return MESH_ALLOCATION_FAILED;
    }

    for (unsigned iblock = 0; iblock < n_blocks; ++iblock)
    {
        const mesh2d_block *block = blocks + iblock;
        const size_t offset = block_offsets[iblock];
        const unsigned n1 = info[iblock].n1;
        const unsigned n2 = info[iblock].n2;

        //  South side of the mesh
        {
            //  South West side
            if (block->bsouth.type == BOUNDARY_TYPE_CURVE)
            {
                res = boundary_point_condition(system_matrix, offset + 0, block->bsouth.curve.x[0],
                                               block->bsouth.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->bwest.type == BOUNDARY_TYPE_CURVE)
            {
                unsigned nb = block->bwest.curve.n;
                res = boundary_point_condition(system_matrix, offset + 0, block->bwest.curve.x[nb - 1],
                                               block->bwest.curve.y[nb - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block *sb = &block->bsouth.block, *wb = &block->bwest.block;
                size_t iw = wb->target;
                size_t offset_wb = block_offsets[iw];
                size_t is = sb->target;
                size_t offset_sb = block_offsets[is];
                res = interior_point_equation(
                    system_matrix, offset + 0, offset_wb + find_boundary_interior_node(info + iw, wb->target_id, 0),
                    offset + 1, offset + n2, offset_sb + find_boundary_interior_node(info + is, sb->target_id, n2 - 1),
                    xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }

            //  South side
            if (block->bsouth.type == BOUNDARY_TYPE_BLOCK)
            {
                const boundary_block *sb = &block->bsouth.block;
                unsigned is = sb->target;
                unsigned offset_sb = block_offsets[is];
                for (unsigned j = 1; j < n2 - 1; ++j)
                {
                    res = interior_point_equation(
                        system_matrix, offset + j, offset + j - 1, offset + j + 1, offset + j + n2,
                        offset_sb + find_boundary_interior_node(info + is, sb->target_id, n2 - 1 - j), xrhs, yrhs, 1);
                    if (res != JMTX_RESULT_SUCCESS)
                    {
                        ret = MESH_MATRIX_FAILURE;
                        goto cleanup_matrix;
                    }
                }
            }
            else
            {
                assert(block->bsouth.type == BOUNDARY_TYPE_CURVE);
                for (unsigned j = 1; j < n2 - 1; ++j)
                {
                    res = boundary_point_condition(system_matrix, offset + j, block->bsouth.curve.x[j],
                                                   block->bsouth.curve.y[j], xrhs, yrhs);
                    if (res != JMTX_RESULT_SUCCESS)
                    {
                        ret = MESH_MATRIX_FAILURE;
                        goto cleanup_matrix;
                    }
                }
            }
            //  South East corner
            if (block->bsouth.type == BOUNDARY_TYPE_CURVE)
            {
                unsigned ns = block->bsouth.n;
                res = boundary_point_condition(system_matrix, offset + n2 - 1, block->bsouth.curve.x[ns - 1],
                                               block->bsouth.curve.y[ns - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->beast.type == BOUNDARY_TYPE_CURVE)
            {
                res = boundary_point_condition(system_matrix, offset + n2 - 1, block->beast.curve.x[0],
                                               block->beast.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block *eb = &block->beast.block, *sb = &block->bsouth.block;
                unsigned ie = eb->target;
                unsigned offset_eb = block_offsets[ie];
                unsigned is = sb->target;
                unsigned offset_sb = block_offsets[is];
                res = interior_point_equation(
                    system_matrix, offset + n2 - 1, offset + n2 - 2,
                    offset_eb + find_boundary_interior_node(info + ie, eb->target_id, n1 - 1), offset + 2 * n2 - 1,
                    offset_sb + find_boundary_interior_node(info + is, sb->target_id, 0), xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        //  Interior of the block
        {
            for (unsigned i = 1; i < n1 - 1; ++i)
            {
                //   West edge
                {
                    unsigned pos = n2 * i + offset;
                    if (block->bwest.type == BOUNDARY_TYPE_CURVE)
                    {
                        res = boundary_point_condition(system_matrix, pos, block->bwest.curve.x[n1 - i - 1],
                                                       block->bwest.curve.y[n1 - i - 1], xrhs, yrhs);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                    else
                    {
                        const boundary_block *wb = &block->bwest.block;
                        unsigned iw = wb->target;
                        unsigned offset_wb = block_offsets[iw];
                        res = interior_point_equation(
                            system_matrix, pos, offset_wb + find_boundary_interior_node(info + iw, wb->target_id, i),
                            pos + 1, pos + n2, pos - n2, xrhs, yrhs, 1);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                }
                //  Interior
                for (unsigned j = 1; j < n2 - 1; ++j)
                {
                    unsigned pos = j + n2 * i + offset;
                    res = interior_point_equation(system_matrix, pos, pos - 1, pos + 1, pos + n2, pos - n2, xrhs, yrhs,
                                                  0);
                    if (res != JMTX_RESULT_SUCCESS)
                    {
                        ret = MESH_MATRIX_FAILURE;
                        goto cleanup_matrix;
                    }
                }
                //   East edge
                {
                    unsigned pos = n2 * i + n2 - 1 + offset;
                    if (block->beast.type == BOUNDARY_TYPE_CURVE)
                    {
                        res = boundary_point_condition(system_matrix, pos, block->beast.curve.x[i],
                                                       block->beast.curve.y[i], xrhs, yrhs);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                    else
                    {
                        const boundary_block *eb = &block->beast.block;
                        unsigned ie = eb->target;
                        unsigned offset_eb = block_offsets[ie];
                        res = interior_point_equation(
                            system_matrix, pos, pos - 1,
                            offset_eb + find_boundary_interior_node(info + ie, eb->target_id, n1 - 1 - i), pos + n2,
                            pos - n2, xrhs, yrhs, 1);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                }
            }
        }

        //   North side
        //       North West corner
        {
            if (block->bnorth.type == BOUNDARY_TYPE_CURVE)
            {
                // valx = block.boundary_n.x[-1]; valy = block.boundary_n.y[-1]
                unsigned nb = block->bnorth.n;
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2, block->bnorth.curve.x[nb - 1],
                                               block->bnorth.curve.y[nb - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->bwest.type == BOUNDARY_TYPE_CURVE)
            {
                // valx = block.boundary_w.x[0]; valy = block.boundary_w.y[0]
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2, block->bwest.curve.x[0],
                                               block->bwest.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block *wb = &block->bwest.block, *nb = &block->bnorth.block;
                unsigned iw = wb->target;
                unsigned offset_wb = block_offsets[iw];
                unsigned in = nb->target;
                unsigned offset_nb = block_offsets[in];
                unsigned pos = offset + (n1 - 1) * n2;
                res = interior_point_equation(
                    system_matrix, pos, offset_wb + find_boundary_interior_node(info + iw, wb->target_id, n1 - 1),
                    offset + (n1 - 1) * n2 + 1, offset_nb + find_boundary_interior_node(info + in, nb->target_id, 0),
                    offset + (n1 - 2) * n2, xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        //   North Side
        if (block->bnorth.type == BOUNDARY_TYPE_BLOCK)
        {
            const boundary_block *nb = &block->bnorth.block;
            unsigned in = nb->target;
            unsigned offset_nb = block_offsets[in];
            for (unsigned j = 1; j < n2 - 1; ++j)
            {
                res = interior_point_equation(system_matrix, offset + (n1 - 1) * n2 + j, offset + (n1 - 1) * n2 + j - 1,
                                              offset + (n1 - 1) * n2 + j + 1,
                                              offset_nb + find_boundary_interior_node(info + in, nb->target_id, j),
                                              offset + (n1 - 1) * n2 + j - n2, xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        else
        {
            for (unsigned j = 1; j < n2 - 1; ++j)
            {
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2 + j,
                                               block->bnorth.curve.x[n2 - 1 - j], block->bnorth.curve.y[n2 - 1 - j],
                                               xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        //       North East corner
        if (block->bnorth.type == BOUNDARY_TYPE_CURVE)
        {
            // valx = block->bnorth.x[0]; valy = block->bnorth.y[0]
            res = boundary_point_condition(system_matrix, offset + n1 * n2 - 1, block->bnorth.curve.x[0],
                                           block->bnorth.curve.y[0], xrhs, yrhs);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
        else if (block->beast.type == BOUNDARY_TYPE_CURVE)
        {
            // valx = block->beast.x[-1]; valy = block->beast.y[-1]
            unsigned nb = block->beast.n;
            res = boundary_point_condition(system_matrix, offset + n1 * n2 - 1, block->beast.curve.x[nb - 1],
                                           block->beast.curve.y[nb - 1], xrhs, yrhs);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
        else
        {
            const boundary_block *eb = &block->beast.block, *nb = &block->bnorth.block;
            unsigned ie = eb->target;
            unsigned offset_eb = block_offsets[ie];
            unsigned in = nb->target;
            unsigned offset_nb = block_offsets[in];
            unsigned pos = offset + n1 * n2 - 1;
            res = interior_point_equation(
                system_matrix, pos, pos - 1, offset_eb + find_boundary_interior_node(info + ie, eb->target_id, 0),
                offset_nb + find_boundary_interior_node(info + in, nb->target_id, n2 - 1), pos - n2, xrhs, yrhs, 1);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
    }

    res = solve_the_system_of_equations(point_cnt, system_matrix, xrhs, yrhs, xnodal, ynodal, allocator,
                                        &allocator_callbacks, cfg, rx, ry);
    if (res != JMTX_RESULT_SUCCESS && res != JMTX_RESULT_NOT_CONVERGED)
    {
        ret = MESH_SOLVER_FAILED;
    }

cleanup_matrix:
    jmtxd_matrix_crs_destroy(system_matrix);
    if (ret != MESH_SUCCESS)
    {
        allocator->free(allocator, ynodal);
        allocator->free(allocator, xnodal);
        allocator->free(allocator, yrhs);
        allocator->free(allocator, xrhs);
        allocator->free(allocator, block_offsets);
        return ret;
    }

    ret = generate_mesh2d_from_geometry(n_blocks, blocks, p_out, allocator,
                                        (mesh_geo_args){
                                            .point_cnt = point_cnt,
                                            .max_lines = max_lines,
                                            .max_surfaces = max_surfaces,
                                            .block_offsets = block_offsets,
                                            .xnodal = xnodal,
                                            .ynodal = ynodal,
                                            .info = info,
                                        }); //, cfg, rx, ry);

    if (ret != MESH_SUCCESS)
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            free_block_info(info + i, allocator);
        }
        allocator->free(allocator, info);
    }
    allocator->free(allocator, ynodal);
    allocator->free(allocator, xnodal);
    allocator->free(allocator, yrhs);
    allocator->free(allocator, xrhs);
    allocator->free(allocator, block_offsets);

    return ret;
}

INTERNAL_MODULE_FUNCTION
void mesh_destroy(mesh2d *mesh, allocator *allocator)
{
    allocator->free(allocator, mesh->p_x);
    allocator->free(allocator, mesh->p_y);
    allocator->free(allocator, mesh->p_lines);
    allocator->free(allocator, mesh->p_surfaces);
    for (unsigned i = 0; i < mesh->n_blocks; ++i)
    {
        free_block_info(mesh->block_info + i, allocator);
    }
    allocator->free(allocator, mesh->block_info);
}

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_lines_info(const mesh2d *mesh, unsigned block, boundary_id boundary,
                                        const geo_id **p_first, unsigned *p_count, int *p_stride)
{
    if (block >= mesh->n_blocks)
    {
        return MESH_INDEX_OUT_OF_BOUNDS;
    }
    unsigned cnt;
    const block_info *const info = mesh->block_info + block;
    switch (boundary)
    {
    case BOUNDARY_ID_EAST:
    case BOUNDARY_ID_WEST:
        cnt = info->n1 - 1;
        break;
    case BOUNDARY_ID_NORTH:
    case BOUNDARY_ID_SOUTH:
        cnt = info->n2 - 1;
        break;
    default:
        return MESH_INVALID_BOUNDARY_ID;
    }
    int first_boundary_index = (int)line_boundary_index(info, boundary, 0);
    int second_boundary_index = (int)line_boundary_index(info, boundary, 1);
    *p_first = info->lines + first_boundary_index;
    *p_stride = second_boundary_index - first_boundary_index;
    *p_count = cnt;
    return MESH_SUCCESS;
}

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_points_info(const mesh2d *mesh, unsigned block, boundary_id boundary,
                                         const geo_id **p_first, unsigned *p_count, int *p_stride)
{
    if (block >= mesh->n_blocks)
    {
        return MESH_INDEX_OUT_OF_BOUNDS;
    }
    unsigned cnt;
    const block_info *const info = mesh->block_info + block;
    switch (boundary)
    {
    case BOUNDARY_ID_EAST:
        cnt = info->n1;
        *p_first = info->points + (info->n2 - 1);
        *p_stride = +info->n2;
        break;
    case BOUNDARY_ID_WEST:
        cnt = info->n1;
        *p_first = info->points + (info->n2 * (info->n1 - 1));
        *p_stride = -info->n2;
        break;
    case BOUNDARY_ID_NORTH:
        cnt = info->n2;
        *p_first = info->points + ((info->n2) * (info->n1) - 1);
        *p_stride = -1;
        break;
    case BOUNDARY_ID_SOUTH:
        cnt = info->n2;
        *p_first = info->points + 0;
        *p_stride = +1;
        break;
    default:
        return MESH_INVALID_BOUNDARY_ID;
    }
    *p_count = cnt;
    return MESH_SUCCESS;
}

INTERNAL_MODULE_FUNCTION
error_id mesh2d_get_boundary_surface_info(const mesh2d *mesh, unsigned block, boundary_id boundary,
                                          const geo_id **p_first, unsigned *p_count, int *p_stride)
{
    if (block >= mesh->n_blocks)
    {
        return MESH_INDEX_OUT_OF_BOUNDS;
    }
    unsigned cnt;
    const block_info *const info = mesh->block_info + block;
    switch (boundary)
    {
    case BOUNDARY_ID_EAST:
        cnt = info->n1 - 1;
        *p_first = info->points + (info->n2 - 2);
        *p_stride = +(info->n2 - 1);
        break;
    case BOUNDARY_ID_WEST:
        cnt = info->n1 - 1;
        *p_first = info->surfaces + ((info->n2 - 1) * (info->n1 - 2));
        *p_stride = -(info->n2 - 1);
        break;
    case BOUNDARY_ID_NORTH:
        cnt = info->n2 - 1;
        *p_first = info->surfaces + ((info->n2 - 1) * (info->n1 - 1) - 1);
        *p_stride = -1;
        break;
    case BOUNDARY_ID_SOUTH:
        cnt = info->n2 - 1;
        *p_first = info->surfaces + 0;
        *p_stride = +1;
        break;
    default:
        return MESH_INVALID_BOUNDARY_ID;
    }
    *p_count = cnt;
    return MESH_SUCCESS;
}
