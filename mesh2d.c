//
// Created by jan on 15.6.2024.
//

#include "mesh2d.h"
#include <jmtx/double/matrices/sparse_row_compressed_safe.h>
#include <jmtx/double/matrices/sparse_column_compressed.h>
#include <jmtx/double/solvers/bicgstab_iteration.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <jmtx/double/matrices/band_row_major_safe.h>
#include <jmtx/double/decompositions/band_lu_decomposition.h>
#include <jmtx/double/decompositions/incomplete_lu_decomposition.h>
#include <jmtx/double/solvers/lu_solving.h>

#include <jmtx/double/matrices/sparse_conversion.h>




static inline unsigned find_boundary_interior_node(const block_info* blk, const boundary_id t, const unsigned idx)
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

static inline jmtx_result interior_point_equation(jmtxd_matrix_crs* mat, const unsigned idx, const  unsigned left,
                                                  const  unsigned right, const unsigned top, const  unsigned btm,
                                                  double* vx, double* vy, const int sort)
{
    uint32_t indices[5] = {btm, left, idx, right, top};
    double values[5] = {1, 1, 1, 1, 1};
    if (sort)
    {
        int needs_sorting = 0;
        for (unsigned n = 0; n < 4; ++n)
        {
            if (indices[n] > indices[n+1])
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

static inline jmtx_result boundary_point_condition(jmtxd_matrix_crs* mat, const unsigned idx, const double x, const double y,
                                                  double* vx, double* vy)
{
    vx[idx] = x;
    vy[idx] = y;
    const uint32_t index = idx;
    const double value = 1;
    return jmtxd_matrix_crs_build_row(mat, idx, 1, &index, &value);
}


enum {DIRECT_SOLVER_LIMIT = (1 << 12), GOD_HELP_ME = (1 << 20), GMRESR_MLIM = (1<<6),
    GCR_TRUNCATION_LIM = (1 << 7)};


static inline jmtx_result solve_the_system_of_equations(const unsigned npts, const jmtxd_matrix_crs* mat,
    double xrhs[const restrict static npts], double yrhs[const restrict static npts],
    double out_x[const restrict npts], double out_y[const restrict npts], allocator* allocator, const jmtx_allocator_callbacks* allocator_callbacks, const solver_config* cfg, double* rx, double* ry)
{
    jmtx_result res = JMTX_RESULT_SUCCESS;
    jmtx_result r1 = JMTX_RESULT_SUCCESS, r2 = JMTX_RESULT_SUCCESS;

    //  Is the problem small enough to solve for it directly?
    if (npts < DIRECT_SOLVER_LIMIT || cfg->direct)
    {
        jmtxd_matrix_brm* banded = NULL, *l = NULL, *u = NULL;
        if (cfg->verbose) printf("Running the direct solver on an %u by %u problem\n", npts, npts);
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

    jmtxd_solver_arguments argsx =
        {
        .in_convergence_criterion = cfg->tol,
        .in_max_iterations = cfg->max_iterations,
        .out_last_error = 1.0,  //  This is here in case we don't run GMRESR
        };
    jmtxd_solver_arguments args_smoother =
        {
        .in_convergence_criterion = 0,
        .in_max_iterations = cfg->smoother_rounds,
        .out_last_error = 1.0,  //  This is here in case we don't run GMRESR
        };
    jmtxd_solver_arguments argsy =
        {
        .in_convergence_criterion = cfg->tol,
        .in_max_iterations = cfg->max_iterations,
        .out_last_error = 1.0,  //  This is here in case we don't run GMRESR
    };

    double* const aux1 = allocator->alloc(allocator, npts * sizeof*aux1);
    double* const aux2 = allocator->alloc(allocator, npts * sizeof*aux2);
    double* const aux3 = allocator->alloc(allocator, npts * sizeof*aux3);
    double* const aux4 = allocator->alloc(allocator, npts * sizeof*aux4);
    double* const aux5 = allocator->alloc(allocator, npts * sizeof*aux5);
    double* const aux6 = allocator->alloc(allocator, npts * sizeof*aux6);
    double* const aux7 = allocator->alloc(allocator, npts * sizeof*aux7);
    double* const aux8 = allocator->alloc(allocator, npts * sizeof*aux8);
    if (!aux1 || !aux2 || !aux3 || !aux4 || !aux5 || !aux6 || !aux7 || !aux8)
    {
        res = JMTX_RESULT_BAD_ALLOC;
        goto end;
    }


    jmtxd_matrix_crs* l, *u;
    jmtxd_matrix_ccs* ccs_u;

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
        if (cfg->verbose) printf("Running PILUBICGSTAB on an %u by %u problem\n", npts, npts);
        r1 = jmtxd_solve_iterative_pilubicgstab_crs(mat, l, u, xrhs, out_x, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, &argsx);
        if (cfg->verbose) printf("PILUBICGSTAB for the x equation finished in %u iterations with an error of %g\n", argsx.out_last_iteration, argsx.out_last_error);
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
        if (cfg->verbose) printf("Running PILUBICGSTAB on an %u by %u problem\n", npts, npts);
        r2 = jmtxd_solve_iterative_pilubicgstab_crs(mat, l, u, yrhs, out_y, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8, &argsy);
        if (cfg->verbose) printf("PILUBICGSTAB for the y equation finished in %u iterations with an error of %g\n", argsy.out_last_iteration, argsy.out_last_error);
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


static inline void free_block_info(block_info* info, allocator* allocator)
{
    allocator->free(allocator, info->points); info->points = NULL;
    allocator->free(allocator, info->lines); info->lines = NULL;
    allocator->free(allocator, info->surfaces); info->surfaces = NULL;
}


static inline unsigned point_boundary_index(const block_info* block, const boundary_id id, const unsigned idx)
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

static inline unsigned point_boundary_index_flipped(const block_info* block, const boundary_id id, const unsigned idx)
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

static inline unsigned line_boundary_index(const block_info* block, const boundary_id id, const unsigned idx)
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

static inline unsigned line_boundary_index_reverse(const block_info* block, const boundary_id id, const unsigned idx)
{
    switch (id)
    {
    case BOUNDARY_ID_SOUTH:
        return ((block->n2 - 1) - idx); //  From 0 to block->n2 - 1
    case BOUNDARY_ID_NORTH:
        return ((block->n1) * (block->n2 - 1) - (idx + 1)); //  From 0 to block->n2 - 1
    case BOUNDARY_ID_WEST:
        return (block->n1 * (block->n2 - 1) + (block->n1 - 1 - idx)); //    From 0 to block->n1 - 1
    case BOUNDARY_ID_EAST:
        return (block->n1 * (block->n2 - 1) + (block->n1 - 1) * (block->n2) + (block->n1 - 1 - idx)); //    From 0 to block->n1 - 1
    }
    return 0;
}

static inline void deal_with_line_boundary(const boundary_block* boundary, const block_info* info_owner, const block_info* info_target)
{
    static const int bnd_map[] =
        {
        [BOUNDARY_ID_NORTH] = 1,
        [BOUNDARY_ID_SOUTH] = -1,
        [BOUNDARY_ID_EAST] = 1,
        [BOUNDARY_ID_WEST] = -1,
        };
    //  When this is 1 we reverse, when it's -1 we do not
    const int reversed = bnd_map[boundary->owner_id] * bnd_map[boundary->target_id];
    if (reversed > 0)
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
    unsigned point_cnt;
    unsigned max_lines;
    unsigned max_surfaces;
    unsigned* block_offsets;
    const double* xnodal;
    const double* ynodal;
    block_info* info;
};

static inline void set_neighboring_block(block_info* bi, boundary_id id, int block_idx)
{
    switch (id)
    {
    case BOUNDARY_ID_NORTH:
        bi->neighboring_block_idx.north = block_idx;
        break;
    case BOUNDARY_ID_SOUTH:
        bi->neighboring_block_idx.south = block_idx;
        break;
    case BOUNDARY_ID_EAST:
        bi->neighboring_block_idx.east = block_idx;
        break;
    case BOUNDARY_ID_WEST:
        bi->neighboring_block_idx.west = block_idx;
        break;
    }
}

static error_id generate_mesh2d_from_geometry(unsigned n_blocks, const mesh2d_block* blocks, mesh2d* p_out, allocator* allocator, const mesh_geo_args args, const solver_config* cfg, double* rx, double* ry)
{
    //  Remove duplicate points by averaging over them
    block_info* info = args.info;
    unsigned unique_pts = 0;
    unsigned* division_factor = allocator->alloc(allocator, args.point_cnt * sizeof*division_factor);
    double* newx = allocator->alloc(allocator, args.point_cnt * sizeof*newx);
    double* newy = allocator->alloc(allocator, args.point_cnt * sizeof*newy);
    line* line_array = allocator->alloc(allocator, args.max_lines * sizeof(*line_array));
    unsigned surf_count = 0;
    surface* surfaces = allocator->alloc(allocator, args.max_surfaces * sizeof*surfaces);
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
        block_info* bi = info + i;
        const mesh2d_block* b = blocks + i;
        unsigned iother;
        int hasn = 0, hass = 0, hase = 0, hasw = 0;
        bi->first_pt = unique_pts;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target)) < i)
        {
            for (unsigned j = 0; j < b->bnorth.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(info + iother, (b->bnorth.block.target_id), j)];
                unsigned this_idx = point_boundary_index(bi, BOUNDARY_ID_NORTH, j);
                newx[other_idx] += args.xnodal[this_idx + args.block_offsets[i]];
                newy[other_idx] += args.ynodal[this_idx + args.block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hasn = 1;
            bi->neighboring_block_idx.north = iother;
            set_neighboring_block(info + iother, b->bnorth.block.target_id, i);
            // duplicate += b->bnorth.n;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target)) < i)
        {
            for (unsigned j = 0; j < b->bsouth.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(info + iother, (b->bsouth.block.target_id), j)];
                unsigned this_idx = point_boundary_index(bi, BOUNDARY_ID_SOUTH, j);
                newx[other_idx] += args.xnodal[this_idx + args.block_offsets[i]];
                newy[other_idx] += args.ynodal[this_idx + args.block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hass = 1;
            bi->neighboring_block_idx.south = iother;
            set_neighboring_block(info + iother, b->bsouth.block.target_id, i);
            // duplicate += b->bsouth.n;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target)) < i)
        {
            for (unsigned j = 0; j < b->beast.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(info + iother, (b->beast.block.target_id), j)];
                unsigned this_idx = point_boundary_index(bi, BOUNDARY_ID_EAST, j);
                newx[other_idx] += args.xnodal[this_idx + args.block_offsets[i]];
                newy[other_idx] += args.ynodal[this_idx + args.block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hase = 1;
            // duplicate += (b->beast.n - (bi->points[bi->n2 - 1] != ~0u) - (bi->points[bi->n2 * bi->n1 - 1] != ~0u));
            bi->neighboring_block_idx.east = iother;
            set_neighboring_block(info + iother, b->beast.block.target_id, i);
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target)) < i)
        {
            for (unsigned j = 0; j < b->bwest.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(info + iother, (b->bwest.block.target_id), j)];
                unsigned this_idx = point_boundary_index(bi, BOUNDARY_ID_WEST, j);
                newx[other_idx] += args.xnodal[this_idx + args.block_offsets[i]];
                newy[other_idx] += args.ynodal[this_idx + args.block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hasw = 1;
            // duplicate += (b->beast.n - (bi->points[0] != ~0u) - (bi->points[bi->n2 * (bi->n1 - 1)] != ~0u));
            bi->neighboring_block_idx.west = iother;
            set_neighboring_block(info + iother, b->bwest.block.target_id, i);
        }
        // unsigned new_pts = bi->n1 * bi->n2 - duplicate;
        // unsigned offset = unique_pts;
        for (unsigned row = hass; row < bi->n1 - hasn; ++row)
        {
            for (unsigned col = hasw; col < bi->n2-hase; ++col)
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
        const double d = 1.0/(double)division_factor[i];
        newx[i] *= d;
        newy[i] *= d;
    }

    {
        double* tmp = realloc(newx, unique_pts * sizeof*newx);
        assert(tmp);
        newx = tmp;
    }
    {
        double* tmp = realloc(newy, unique_pts * sizeof*newy);
        assert(tmp);
        newy = tmp;
    }
    allocator->free(allocator, division_factor);



    // Create mesh line info
    unsigned line_count = 1;
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info* bi = info + i;
        const mesh2d_block* b = blocks + i;
        unsigned hasn = 0, hass = 0, hase = 0, hasw = 0;
        unsigned iother;
        bi->first_ln = line_count;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target)) < i)
        {
            deal_with_line_boundary(&b->bnorth.block, info + i, info + iother);
            hasn = 1;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target)) < i)
        {
            deal_with_line_boundary(&b->bsouth.block, info + i, info + iother);
            hass = 1;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target)) < i)
        {
            deal_with_line_boundary(&b->beast.block, info + i, info + iother);
            hase = 1;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target)) < i)
        {
            deal_with_line_boundary(&b->bwest.block, info + i, info + iother);
            hasw = 1;
        }

        for (unsigned row = hass; row < bi->n1 - hasn; ++row)
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

        for (unsigned col = hasw; col < bi->n2 - hase; ++col)
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
        for (unsigned j = 0; j < (bi->n2 - 1) * bi->n1 + bi->n2 * (bi->n1 - 1); ++j)
        {
            assert(bi->lines[i] != 0);
        }
        bi->last_ln = line_count;
    }
    line_count -= 1;

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info* bi = info + i;
        for (unsigned row = 0; row < bi->n1 - 1; ++row)
        {
            for (unsigned col = 0; col < bi->n2 - 1; ++col)
            {
                geo_id btm = (geo_id)(bi->lines[col + row * (bi->n2 - 1)]);
                geo_id top = (geo_id)(bi->lines[col + (row + 1) * (bi->n2 - 1)]);
                geo_id lft = (geo_id)(bi->lines[bi->n1 * (bi->n2 - 1) + row + col * (bi->n1 - 1)]);
                geo_id rgt = (geo_id)(bi->lines[bi->n1 * (bi->n2 - 1) + row + (col + 1) * (bi->n1 - 1)]);
                assert(surf_count < args.max_surfaces);
                bi->surfaces[col + row*(bi->n2 - 1)] = (geo_id)surf_count;
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

static inline error_id check_boundary_consistency(const mesh2d_block* blocks, const boundary_block* bnd, const unsigned idx)
{
    const boundary* other = NULL;
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
        return MESH_BOUNDARY_POINT_COUNT_MISMATCH;
    }
    if (other->type == BOUNDARY_TYPE_CURVE && bnd->target >= idx)
    {
        return MESH_BOUNDARY_UNSORTED;
    }
    return MESH_SUCCESS;
}

error_id mesh2d_check_blocks(unsigned n_blocks, const mesh2d_block* blocks)
{
    error_id ret = MESH_SUCCESS;
    for (unsigned iblk = 0; iblk < n_blocks; ++iblk)
    {
        const mesh2d_block* b = blocks + iblk;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (ret = check_boundary_consistency(blocks, &b->bnorth.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (ret = check_boundary_consistency(blocks, &b->bsouth.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (ret = check_boundary_consistency(blocks, &b->beast.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (ret = check_boundary_consistency(blocks, &b->bwest.block, iblk)) != MESH_SUCCESS)
        {
            break;
        }
    }
    return ret;
}

error_id mesh2d_create_elliptical(unsigned n_blocks, const mesh2d_block* blocks, const solver_config* cfg, allocator* allocator, mesh2d* p_out, double* rx, double* ry)
{
    jmtx_allocator_callbacks allocator_callbacks =
        {
            .alloc = allocator->alloc,
            .free = allocator->free,
            .realloc = allocator->realloc,
            .state = (void*)0xB16B00B135,
        };
    error_id ret = MESH_SUCCESS;
    unsigned point_cnt = 0;
    unsigned max_lines = 0;
    unsigned max_surfaces = 0;
    unsigned* block_offsets = allocator->alloc(allocator, n_blocks * sizeof*block_offsets);
    block_info* info = allocator->alloc(allocator, n_blocks * sizeof*info);
    if (!block_offsets || !info)
    {
        allocator->free(allocator, block_offsets);
        allocator->free(allocator, info);
        return MESH_ALLOCATION_FAILED;
    }
    memset(block_offsets, 0, n_blocks * sizeof(*block_offsets));
    for (unsigned iblk = 0; iblk < n_blocks; ++iblk)
    {
        const mesh2d_block* const blk = blocks + iblk;
        //  Check mesh boundaries and compute each block's size
        unsigned nnorth = blk->bnorth.n;
        unsigned nsouth = blk->bsouth.n;
        if (nnorth != nsouth)
        {
            return MESH_BOUNDARY_SIZE_MISMATCH;
        }
        unsigned neast = blk->beast.n;
        unsigned nwest = blk->bwest.n;
        if (neast != nwest)
        {
            return MESH_BOUNDARY_SIZE_MISMATCH;
        }

        unsigned npts = nnorth * neast;
        unsigned n1 = neast;
        unsigned n2 = nnorth;
        info[iblk].n1 = n1;
        info[iblk].n2 = n2;
        info[iblk].points = allocator->alloc(allocator, n1 * n2 * sizeof(*info[iblk].points));
        assert(info[iblk].points);
        memset(info[iblk].points, ~0u, n1 * n2 * sizeof(*info[iblk].points));
        info[iblk].lines = allocator->alloc(allocator, (n1 * (n2 - 1) + (n1 - 1) * n2) * sizeof(*info[iblk].lines));
        assert(info[iblk].lines);
        info[iblk].surfaces = allocator->alloc(allocator, (n1 - 1) * (n2 - 1) * sizeof(*info[iblk].surfaces));
        assert(info[iblk].surfaces);
        info[iblk].neighboring_block_idx.north = -1;
        info[iblk].neighboring_block_idx.south = -1;
        info[iblk].neighboring_block_idx.east = -1;
        info[iblk].neighboring_block_idx.west = -1;
        if (iblk != n_blocks - 1)
        {
            block_offsets[iblk + 1] = npts + block_offsets[iblk];
        }
        point_cnt += npts;
        max_lines += (n1 - 1) * n2 + n1 * (n2 - 1);
        max_surfaces += (n1 - 1) * (n2 - 1);
    }

    double* xrhs = allocator->alloc(allocator, point_cnt * sizeof*xrhs);
    double* yrhs = allocator->alloc(allocator, point_cnt * sizeof*yrhs);
    double* xnodal = allocator->alloc(allocator, point_cnt * sizeof*xnodal);
    double* ynodal = allocator->alloc(allocator, point_cnt * sizeof*ynodal);
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
    jmtxd_matrix_crs* system_matrix;
    jmtx_result res = jmtxds_matrix_crs_new(&system_matrix, point_cnt, point_cnt, 4*point_cnt, &allocator_callbacks);
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
        const mesh2d_block* block = blocks + iblock;
        const unsigned offset = block_offsets[iblock];
        const unsigned n1 = info[iblock].n1;
        const unsigned n2 = info[iblock].n2;

        //  South side of the mesh
        {
            //  South West side
            if (block->bsouth.type == BOUNDARY_TYPE_CURVE)
            {
                res = boundary_point_condition(system_matrix, offset + 0, block->bsouth.curve.x[0], block->bsouth.curve.y[0],
                                        xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->bwest.type == BOUNDARY_TYPE_CURVE)
            {
                unsigned nb = block->bwest.curve.n;
                res = boundary_point_condition(system_matrix, offset + 0, block->bwest.curve.x[nb-1], block->bwest.curve.y[nb-1],
                                              xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block* sb = &block->bsouth.block, *wb = &block->bwest.block;
                unsigned iw = wb->target;
                unsigned offset_wb = block_offsets[iw];
                unsigned is = sb->target;
                unsigned offset_sb = block_offsets[is];
                res = interior_point_equation(system_matrix, offset + 0, offset_wb +
                find_boundary_interior_node(info + iw, wb->target_id, 0), offset + 1, offset + n2,
                offset_sb + find_boundary_interior_node(info + is, sb->target_id, n2 - 1), xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }

            //  South side
            if (block->bsouth.type == BOUNDARY_TYPE_BLOCK)
            {
                const boundary_block* sb = &block->bsouth.block;
                unsigned is = sb->target;
                unsigned offset_sb = block_offsets[is];
                for (unsigned j = 1; j < n2 - 1; ++j)
                {
                    res = interior_point_equation(system_matrix, offset + j, offset + j - 1, offset + j + 1, offset + j + n2, offset_sb + find_boundary_interior_node(info + is, sb->target_id, n2 - 1 - j), xrhs, yrhs, 1);
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
                    res = boundary_point_condition(system_matrix, offset + j, block->bsouth.curve.x[j], block->bsouth.curve.y[j], xrhs, yrhs);
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
                res = boundary_point_condition(system_matrix, offset + n2 - 1, block->bsouth.curve.x[ns - 1], block->bsouth.curve.y[ns - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->beast.type == BOUNDARY_TYPE_CURVE)
            {
                res = boundary_point_condition(system_matrix, offset + n2 - 1, block->beast.curve.x[0], block->beast.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block* eb = &block->beast.block, * sb = &block->bsouth.block;
                unsigned ie = eb->target;
                unsigned offset_eb = block_offsets[ie];
                unsigned is = sb->target;
                unsigned offset_sb = block_offsets[is];
                res = interior_point_equation(system_matrix, offset + n2-1, offset + n2-2, offset_eb + find_boundary_interior_node(info + ie, eb->target_id, n1 - 1), offset + 2*n2-1, offset_sb + find_boundary_interior_node(info + is, sb->target_id, 0), xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        //  Interior of the block
        {
            for (unsigned i = 1; i < n1-1; ++i)
            {
                //   West edge
                {
                    unsigned pos = n2 * i  + offset;
                    if (block->bwest.type == BOUNDARY_TYPE_CURVE)
                    {
                        res = boundary_point_condition(system_matrix, pos, block->bwest.curve.x[n1-i-1], block->bwest.curve.y[n1-i-1], xrhs, yrhs);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                    else
                    {
                        const boundary_block* wb = &block->bwest.block;
                        unsigned iw = wb->target;
                        unsigned offset_wb = block_offsets[iw];
                        res = interior_point_equation(system_matrix, pos, offset_wb + find_boundary_interior_node(info + iw, wb->target_id, i), pos + 1, pos + n2, pos - n2, xrhs, yrhs, 1);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                }
                //  Interior
                for (unsigned j = 1; j < n2-1; ++j)
                {
                    unsigned pos = j + n2 * i  + offset;
                    res = interior_point_equation(system_matrix, pos, pos-1, pos+1, pos+n2, pos-n2, xrhs, yrhs, 0);
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
                        res = boundary_point_condition(system_matrix, pos, block->beast.curve.x[i], block->beast.curve.y[i], xrhs, yrhs);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                    else
                    {
                        const boundary_block* eb = &block->beast.block;
                        unsigned ie = eb->target;
                        unsigned offset_eb = block_offsets[ie];
                        res = interior_point_equation(system_matrix, pos, pos - 1, offset_eb + find_boundary_interior_node(info + ie, eb->target_id, n1 - 1 - i), pos + n2, pos - n2, xrhs, yrhs, 1);
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
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2, block->bnorth.curve.x[nb - 1], block->bnorth.curve.y[nb - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->bwest.type == BOUNDARY_TYPE_CURVE)
            {
                // valx = block.boundary_w.x[0]; valy = block.boundary_w.y[0]
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2, block->bwest.curve.x[0], block->bwest.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block* wb = &block->bwest.block, *nb = &block->bnorth.block;
                unsigned iw = wb->target;
                unsigned offset_wb = block_offsets[iw];
                unsigned in = nb->target;
                unsigned offset_nb = block_offsets[in];
                unsigned pos = offset + (n1 - 1) * n2;
                res = interior_point_equation(system_matrix, pos, offset_wb + find_boundary_interior_node(info + iw, wb->target_id, n1 - 1), offset + (n1 - 1) * n2 + 1, offset_nb + find_boundary_interior_node(info + in, nb->target_id, 0), offset + (n1 - 2) * n2, xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }

        }
        //   Noth Side
        if (block->bnorth.type == BOUNDARY_TYPE_BLOCK)
        {
            const boundary_block* nb= &block->bnorth.block;
            unsigned in = nb->target;
            unsigned offset_nb = block_offsets[in];
            for (unsigned j = 1; j < n2 - 1; ++j)
            {
                res = interior_point_equation(system_matrix, offset + (n1 - 1) * n2 + j, offset + (n1 - 1) * n2 + j - 1, offset + (n1 - 1) * n2 + j + 1, offset_nb + find_boundary_interior_node(info + in, nb->target_id, j), offset + (n1 - 1) * n2 + j - n2, xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        else
        {
            for (unsigned j  = 1; j < n2 - 1; ++j)
            {
                res = boundary_point_condition(system_matrix, offset + (n1 - 1) * n2 + j, block->bnorth.curve.x[n2 - 1 - j], block->bnorth.curve.y[n2 - 1 - j], xrhs, yrhs);
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
            res = boundary_point_condition(system_matrix, offset + n1 * n2 - 1, block->bnorth.curve.x[0], block->bnorth.curve.y[0], xrhs, yrhs);
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
            res = boundary_point_condition(system_matrix, offset + n1 * n2 - 1, block->beast.curve.x[nb - 1], block->beast.curve.y[nb - 1], xrhs, yrhs);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
        else
        {
            const boundary_block* eb = &block->beast.block, *nb = &block->bnorth.block;
            unsigned ie = eb->target;
            unsigned offset_eb = block_offsets[ie];
            unsigned in = nb->target;
            unsigned offset_nb = block_offsets[in];
            unsigned pos = offset + n1 * n2 - 1;
            res = interior_point_equation(system_matrix, pos, pos - 1, offset_eb + find_boundary_interior_node(info + ie, eb->target_id, 0), offset_nb + find_boundary_interior_node(info + in, nb->target_id, n2 - 1), pos - n2, xrhs, yrhs, 1);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
    }

    res = solve_the_system_of_equations(point_cnt, system_matrix, xrhs, yrhs, xnodal, ynodal, allocator, &allocator_callbacks, cfg, rx, ry);
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
        }, cfg, rx, ry);
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


void mesh_destroy(mesh2d* mesh, allocator* allocator)
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
