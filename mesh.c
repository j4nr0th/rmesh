//
// Created by jan on 15.6.2024.
//

#include "mesh.h"
#include <jmtx/double/matrices/sparse_row_compressed_safe.h>
#include <jmtx/double/solvers/bicgstab_iteration.h>
#include <jmtx/double/solvers/gauss_seidel_iteration.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <jmtx/double/matrices/band_row_major_safe.h>
#include <jmtx/double/decompositions/band_lu_decomposition.h>
#include <jmtx/double/solvers/lu_solving.h>

#include <jmtx/double/matrices/sparse_conversion.h>

#include "jmtx/double/solvers/recursive_generalized_minimum_residual_iteration.h"


static inline unsigned find_boundary_interior_node(const mesh_block* blk, boundary_id t, unsigned idx)
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

static inline jmtx_result interior_point_equation(jmtxd_matrix_crs* mat, unsigned idx, unsigned left, unsigned right,
                                                  unsigned top, unsigned btm, double* vx, double* vy, int sort)
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
    return jmtxds_matrix_crs_set_row(mat, idx, 5, indices, values);
}

static inline jmtx_result boundary_point_condition(jmtxd_matrix_crs* mat, unsigned idx, double x, double y,
                                                  double* vx, double* vy)
{
    vx[idx] = x;
    vy[idx] = y;
    uint32_t index = idx;
    double value = 1;
    return jmtxds_matrix_crs_set_row(mat, idx, 1, &index, &value);
}


enum {DIRECT_SOLVER_LIMIT = (1 << 12), GMRESR_SOLVER_LIMIT = (1 << 18), GOD_HELP_ME = (1 << 20), GMRESR_MLIM = (1<<6),
    GCR_TRUNCATION_LIM = (1 << 7)};


static inline jmtx_result solve_the_system_of_equations(unsigned npts, jmtxd_matrix_crs* mat,
    double xrhs[const restrict static npts], double yrhs[const restrict static npts],
    double out_x[const restrict npts], double out_y[const restrict npts])
{
    jmtxd_matrix_brm* banded = NULL, *l = NULL, *u = NULL;
    jmtx_result res = JMTX_RESULT_SUCCESS;
    jmtx_result r1 = JMTX_RESULT_SUCCESS, r2 = JMTX_RESULT_SUCCESS;

    //  Is the problem small enough to solve for it directly?
    if (npts < DIRECT_SOLVER_LIMIT)
    {
        printf("Running the direct solver on an %u by %u problem\n", npts, npts);
        res = jmtxd_convert_crs_to_brm(mat, &banded, NULL);
        if (res != JMTX_RESULT_SUCCESS)
        {
            return res;
        }
        res = jmtxd_decompose_lu_brm(banded, &l, &u, NULL);
        jmtxd_matrix_brm_destroy(banded);
        if (res != JMTX_RESULT_SUCCESS)
        {
            return res;
        }
        jmtxd_solve_direct_lu_brm(l, u, xrhs, out_x);
        jmtxd_solve_direct_lu_brm(l, u, yrhs, out_y);
        jmtxd_matrix_brm_destroy(l);
        jmtxd_matrix_brm_destroy(u);
        return JMTX_RESULT_SUCCESS;
    }

    jmtxd_solver_arguments argsx =
        {
        .in_convergence_criterion = 1e-9,
        .in_max_iterations = 1000,
        .out_last_error = 1.0,  //  This is here in case we don't run GMRESR
        };
    jmtxd_solver_arguments argsy =
        {
        .in_convergence_criterion = 1e-9,
        .in_max_iterations = 1000,
        .out_last_error = 1.0,  //  This is here in case we don't run GMRESR
    };

    double* const aux1 = calloc(npts, sizeof*aux1);
    assert(aux1);
    double* const aux2 = calloc(npts, sizeof*aux2);
    assert(aux2);
    double* const aux3 = calloc(npts, sizeof*aux3);
    assert(aux3);
    double* const aux4 = calloc(npts, sizeof*aux4);
    assert(aux4);
    double* const aux5 = calloc(npts, sizeof*aux5);
    assert(aux5);
    double* const aux6 = calloc(npts, sizeof*aux6);
    assert(aux6);
    double* const auxvecs1 = calloc(npts*GMRESR_MLIM, sizeof*auxvecs1);
    assert(auxvecs1);
    double* const auxvecs2 = calloc(npts*GCR_TRUNCATION_LIM, sizeof*auxvecs2);
    assert(auxvecs2);
    double* const auxvecs3 = calloc(npts*GCR_TRUNCATION_LIM, sizeof*auxvecs3);
    assert(auxvecs3);

    if (npts < GMRESR_SOLVER_LIMIT)
    {
        printf("Running GMRESR on an %u by %u problem\n", npts, npts);
        jmtxd_matrix_brm* aux_mat_gmresr;
        res = jmtxd_matrix_brm_new(&aux_mat_gmresr, GMRESR_MLIM, GMRESR_MLIM, GMRESR_MLIM-1, 0, NULL, NULL);
        if (res != JMTX_RESULT_SUCCESS)
        {
            goto end;
        }
        argsx.in_max_iterations = 64;
        jmtxd_solve_iterative_gmresr_crs(mat, xrhs, out_x, GMRESR_MLIM, GCR_TRUNCATION_LIM, aux_mat_gmresr,
        aux1, aux2, aux3, aux4, aux5, aux6, auxvecs1, auxvecs2, auxvecs3, &argsx);
        printf("GMRESR for the x equation finished in %u iterations with an error of %g\n", argsx.out_last_iteration, argsx.out_last_error);
        if (!isfinite(argsx.out_last_error))
        {
            argsx.out_last_error = 1;
            memset(out_x, 0, npts*sizeof(*out_x));
        }
        argsx.in_max_iterations = 1024;
        argsy.in_max_iterations = 64;
        jmtxd_solve_iterative_gmresr_crs(mat, yrhs, out_y, GMRESR_MLIM, GCR_TRUNCATION_LIM, aux_mat_gmresr,
                aux1, aux2, aux3, aux4, aux5, aux6, auxvecs1, auxvecs2, auxvecs3, &argsy);
        printf("GMRESR for the y equation finished in %u iterations with an error of %g\n", argsx.out_last_iteration, argsx.out_last_error);
        if (!isfinite(argsy.out_last_error))
        {
            argsy.out_last_error = 1;
            memset(out_y, 0, npts*sizeof(*out_y));
        }
        argsx.in_max_iterations = 1024;

        jmtxd_matrix_brm_destroy(aux_mat_gmresr);
    }

    if (argsx.out_last_error > argsx.in_convergence_criterion)
    {
        printf("Running BICG-Stab on an %u by %u problem\n", npts, npts);
        r1 = jmtxd_solve_iterative_bicgstab_crs(mat, xrhs, out_x, aux1, aux2, aux3, aux4, aux5, aux6, &argsx);
        printf("BICG-Stab for the x equation finished in %u iterations with an error of %g\n", argsx.out_last_iteration, argsx.out_last_error);
    }
    if (argsy.out_last_error > argsy.in_convergence_criterion)
    {
        printf("Running BICG-Stab on an %u by %u problem\n", npts, npts);
        r2 = jmtxd_solve_iterative_bicgstab_crs(mat, xrhs, out_x, aux1, aux2, aux3, aux4, aux5, aux6, &argsy);
        printf("BICG-Stab for the y equation finished in %u iterations with an error of %g\n", argsx.out_last_iteration, argsx.out_last_error);
    }

end:
    free(auxvecs3);
    free(auxvecs2);
    free(auxvecs1);
    free(aux6);
    free(aux5);
    free(aux4);
    free(aux3);
    free(aux2);
    free(aux1);

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


static inline void free_block_info(block_info* info)
{
    free(info->points); info->points = NULL;
    free(info->lines); info->lines = NULL;
    free(info->surfaces); info->surfaces = NULL;
}


static inline unsigned point_boundary_index(const mesh_block* block, boundary_id id, unsigned idx)
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

static inline unsigned point_boundary_index_flipped(const mesh_block* block, boundary_id id, unsigned idx)
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

static inline unsigned line_boundary_index(const mesh_block* block, boundary_id id, unsigned idx)
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

static inline unsigned line_boundary_index_reverse(const mesh_block* block, boundary_id id, unsigned idx)
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
    const int reversed = bnd_map[boundary->id1] * bnd_map[boundary->target_id];
    if (reversed > 0)
    {
        for (unsigned i = 0; i < boundary->n - 1; ++i)
        {
            geo_id iother = -info_target->lines[line_boundary_index_reverse(boundary->target, boundary->target_id, i)];
            unsigned this_idx = line_boundary_index(boundary->b1, boundary->id1, i);
            info_owner->lines[this_idx] = iother;
        }
    }
    else
    {
        for (unsigned i = 0; i < boundary->n - 1; ++i)
        {
            geo_id iother = info_target->lines[line_boundary_index(boundary->target, boundary->target_id, i)];
            unsigned this_idx = line_boundary_index(boundary->b1, boundary->id1, i);
            info_owner->lines[this_idx] = iother;
        }
    }
}


error_id mesh_create(unsigned int n_blocks, mesh_block* blocks, mesh* p_out)
{
    error_id ret = MESH_SUCCESS;
    unsigned point_cnt = 0;
    unsigned max_lines = 0;
    unsigned max_surfaces = 0;
    unsigned* block_offsets = calloc(n_blocks, sizeof*block_offsets);
    assert(block_offsets);
    memset(block_offsets, 0, n_blocks * sizeof(*block_offsets));
    for (unsigned iblk = 0; iblk < n_blocks; ++iblk)
    {
        mesh_block* const blk = blocks + iblk;
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
        blk->n1 = neast;
        blk->n2 = nnorth;
        blk->npts = npts;
        if (iblk != n_blocks - 1)
        {
            block_offsets[iblk + 1] = npts + block_offsets[iblk];
        }
        point_cnt += npts;
        max_lines += (blk->n1 - 1) * blk->n2 + blk->n1 * (blk->n2 - 1);
        max_surfaces += (blk->n1 - 1) * (blk->n2 - 1);
    }

    double* xrhs = calloc(point_cnt, sizeof*xrhs);
    assert(xrhs);
    double* yrhs = calloc(point_cnt, sizeof*yrhs);
    assert(yrhs);
    double* xnodal = calloc(point_cnt, sizeof*xnodal);
    assert(xnodal);
    double* ynodal = calloc(point_cnt, sizeof*ynodal);
    assert(ynodal);

    jmtxd_matrix_crs* system_matrix;
    jmtx_result res = jmtxds_matrix_crs_new(&system_matrix, point_cnt, point_cnt, 4*point_cnt, NULL);
    if (res != JMTX_RESULT_SUCCESS)
    {
        free(ynodal);
        free(xnodal);
        free(yrhs);
        free(xrhs);
        free(block_offsets);
        return MESH_ALLOCATION_FAILED;
    }

    for (unsigned iblock = 0; iblock < n_blocks; ++iblock)
    {
        const mesh_block* block = blocks + iblock;
        const unsigned offset = block_offsets[iblock];

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
                unsigned offset_wb = block_offsets[wb->target - blocks];
                unsigned offset_sb = block_offsets[sb->target - blocks];
                res = interior_point_equation(system_matrix, offset + 0, offset_wb +
                find_boundary_interior_node(wb->target, wb->target_id, 0), offset + 1, offset + block->n2,
                offset_sb + find_boundary_interior_node(sb->target, sb->target_id, block->n2 - 1), xrhs, yrhs, 1);
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
                unsigned offset_sb = block_offsets[sb->target - blocks];
                for (unsigned j = 1; j < block->n2 - 1; ++j)
                {
                    res = interior_point_equation(system_matrix, offset + j, offset + j - 1, offset + j + 1, offset + j + block->n2, offset_sb + find_boundary_interior_node(sb->target, sb->target_id, block->n2 - 1 - j), xrhs, yrhs, 1);
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
                for (unsigned j = 1; j < block->n2 - 1; ++j)
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
                res = boundary_point_condition(system_matrix, offset + block->n2 - 1, block->bsouth.curve.x[ns - 1], block->bsouth.curve.y[ns - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->beast.type == BOUNDARY_TYPE_CURVE)
            {
                res = boundary_point_condition(system_matrix, offset + block->n2 - 1, block->beast.curve.x[0], block->beast.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block* eb = &block->beast.block, * sb = &block->bsouth.block;
                unsigned offset_eb = block_offsets[eb->target - blocks];
                unsigned offset_sb = block_offsets[sb->target - blocks];
                res = interior_point_equation(system_matrix, offset + block->n2-1, offset + block->n2-2, offset_eb + find_boundary_interior_node(eb->target, eb->target_id, block->n1 - 1), offset + 2*block->n2-1, offset_sb + find_boundary_interior_node(sb->target, sb->target_id, 0), xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        //  Interior of the block
        {
            for (unsigned i = 1; i < block->n1-1; ++i)
            {
                //   West edge
                {
                    unsigned pos = block->n2 * i  + offset;
                    if (block->bwest.type == BOUNDARY_TYPE_CURVE)
                    {
                        res = boundary_point_condition(system_matrix, pos, block->bwest.curve.x[block->n1-i-1], block->bwest.curve.y[block->n1-i-1], xrhs, yrhs);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                    else
                    {
                        const boundary_block* wb = &block->bwest.block;
                        unsigned offset_wb = block_offsets[wb->target - blocks];
                        res = interior_point_equation(system_matrix, pos, offset_wb + find_boundary_interior_node(wb->target, wb->target_id, i), pos + 1, pos + block->n2, pos - block->n2, xrhs, yrhs, 1);
                        if (res != JMTX_RESULT_SUCCESS)
                        {
                            ret = MESH_MATRIX_FAILURE;
                            goto cleanup_matrix;
                        }
                    }
                }
                //  Interior
                for (unsigned j = 1; j < block->n2-1; ++j)
                {
                    unsigned pos = j + block->n2 * i  + offset;
                    res = interior_point_equation(system_matrix, pos, pos-1, pos+1, pos+block->n2, pos-block->n2, xrhs, yrhs, 0);
                    if (res != JMTX_RESULT_SUCCESS)
                    {
                        ret = MESH_MATRIX_FAILURE;
                        goto cleanup_matrix;
                    }
                }
                //   East edge
                {
                    unsigned pos = block->n2 * i + block->n2 - 1 + offset;
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
                        unsigned offset_eb = block_offsets[eb->target - blocks];
                        res = interior_point_equation(system_matrix, pos, pos - 1, offset_eb + find_boundary_interior_node(eb->target, eb->target_id, block->n1 - 1 - i), pos + block->n2, pos - block->n2, xrhs, yrhs, 1);
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
                res = boundary_point_condition(system_matrix, offset + (block->n1 - 1) * block->n2, block->bnorth.curve.x[nb - 1], block->bnorth.curve.y[nb - 1], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else if (block->bwest.type == BOUNDARY_TYPE_CURVE)
            {
                // valx = block.boundary_w.x[0]; valy = block.boundary_w.y[0]
                res = boundary_point_condition(system_matrix, offset + (block->n1 - 1) * block->n2, block->bwest.curve.x[0], block->bwest.curve.y[0], xrhs, yrhs);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
            else
            {
                const boundary_block* wb = &block->bwest.block, *nb = &block->bnorth.block;
                unsigned offset_wb = block_offsets[wb->target - blocks];
                unsigned offset_nb = block_offsets[nb->target - blocks];
                unsigned pos = offset + (block->n1 - 1) * block->n2;
                res = interior_point_equation(system_matrix, pos, offset_wb + find_boundary_interior_node(wb->target, wb->target_id, block->n1 - 1), offset + (block->n1 - 1) * block->n2 + 1, offset_nb + find_boundary_interior_node(nb->target, nb->target_id, 0), offset + (block->n1 - 2) * block->n2, xrhs, yrhs, 1);
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
            unsigned offset_nb = block_offsets[nb->target - blocks];
            for (unsigned j = 1; j < block->n2 - 1; ++j)
            {
                res = interior_point_equation(system_matrix, offset + (block->n1 - 1) * block->n2 + j, offset + (block->n1 - 1) * block->n2 + j - 1, offset + (block->n1 - 1) * block->n2 + j + 1, offset_nb + find_boundary_interior_node(nb->target, nb->target_id, j), offset + (block->n1 - 1) * block->n2 + j - block->n2, xrhs, yrhs, 1);
                if (res != JMTX_RESULT_SUCCESS)
                {
                    ret = MESH_MATRIX_FAILURE;
                    goto cleanup_matrix;
                }
            }
        }
        else
        {
            for (unsigned j  = 1; j < block->n2 - 1; ++j)
            {
                res = boundary_point_condition(system_matrix, offset + (block->n1 - 1) * block->n2 + j, block->bnorth.curve.x[block->n2 - 1 - j], block->bnorth.curve.y[block->n2 - 1 - j], xrhs, yrhs);
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
            res = boundary_point_condition(system_matrix, offset + block->n1 * block->n2 - 1, block->bnorth.curve.x[0], block->bnorth.curve.y[0], xrhs, yrhs);
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
            res = boundary_point_condition(system_matrix, offset + block->n1 * block->n2 - 1, block->beast.curve.x[nb - 1], block->beast.curve.y[nb - 1], xrhs, yrhs);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
        else
        {
            const boundary_block* eb = &block->beast.block, *nb = &block->bnorth.block;
            unsigned offset_eb = block_offsets[eb->target - blocks];
            unsigned offset_nb = block_offsets[nb->target - blocks];
            unsigned pos = offset + block->n1 * block->n2 - 1;
            res = interior_point_equation(system_matrix, pos, pos - 1, offset_eb + find_boundary_interior_node(eb->target, eb->target_id, 0), offset_nb + find_boundary_interior_node(nb->target, nb->target_id, block->n2 - 1), pos - block->n2, xrhs, yrhs, 1);
            if (res != JMTX_RESULT_SUCCESS)
            {
                ret = MESH_MATRIX_FAILURE;
                goto cleanup_matrix;
            }
        }
    }

    res = solve_the_system_of_equations(point_cnt, system_matrix, xrhs, yrhs, xnodal, ynodal);
    if (res != JMTX_RESULT_SUCCESS)
    {
        ret = MESH_SOLVER_FAILED;
    }

cleanup_matrix:
    jmtxd_matrix_crs_destroy(system_matrix);
    if (ret != MESH_SUCCESS)
    {
        free(ynodal);
        free(xnodal);
        free(yrhs);
        free(xrhs);
        free(block_offsets);
        return ret;
    }

    //  Remove duplicate points by averaging over them
    block_info* info = calloc(n_blocks, sizeof*info);
    assert(info);
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        info[i].n1 = blocks[i].n1;
        info[i].n2 = blocks[i].n2;
        info[i].points = calloc(blocks[i].n1 * blocks[i].n2, sizeof(*info[i].points));
        assert(info[i].points);
        memset(info[i].points, ~0u, blocks[i].n1 * blocks[i].n2 * sizeof(*info[i].points));
        info[i].lines = calloc(blocks[i].n1 * (blocks[i].n2 - 1) + (blocks[i].n1 - 1) * blocks[i].n2, sizeof(*info[i].lines));
        assert(info[i].lines);
        info[i].surfaces = calloc((blocks[i].n1 - 1) * (blocks[i].n2 - 1), sizeof(*info[i].surfaces));
        assert(info[i].surfaces);
        info[i].neighboring_block_idx.north = -1;
        info[i].neighboring_block_idx.south = -1;
        info[i].neighboring_block_idx.east = -1;
        info[i].neighboring_block_idx.west = -1;
    }
    unsigned unique_pts = 0;
    unsigned* division_factor = calloc(point_cnt, sizeof*division_factor);
    assert(division_factor);
    double* newx = calloc(point_cnt, sizeof*newx);
    double* newy = calloc(point_cnt, sizeof*newy);
    assert(newx);
    assert(newy);

    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info* bi = info + i;
        const mesh_block* b = blocks + i;
        unsigned iother;
        int hasn = 0, hass = 0, hase = 0, hasw = 0;
        bi->first_pt = unique_pts;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target - blocks)) < i)
        {
            for (unsigned j = 0; j < b->bnorth.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(b->bnorth.block.target, (b->bnorth.block.target_id), j)];
                unsigned this_idx = point_boundary_index(b, BOUNDARY_ID_NORTH, j);
                newx[other_idx] += xnodal[this_idx + block_offsets[i]];
                newy[other_idx] += ynodal[this_idx + block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hasn = 1;
            bi->neighboring_block_idx.north = iother;
            // duplicate += b->bnorth.n;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target - blocks)) < i)
        {
            for (unsigned j = 0; j < b->bsouth.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(b->bsouth.block.target, (b->bsouth.block.target_id), j)];
                unsigned this_idx = point_boundary_index(b, BOUNDARY_ID_SOUTH, j);
                newx[other_idx] += xnodal[this_idx + block_offsets[i]];
                newy[other_idx] += ynodal[this_idx + block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hass = 1;
            bi->neighboring_block_idx.south = iother;
            // duplicate += b->bsouth.n;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target - blocks)) < i)
        {
            for (unsigned j = 0; j < b->beast.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(b->beast.block.target, (b->beast.block.target_id), j)];
                unsigned this_idx = point_boundary_index(b, BOUNDARY_ID_EAST, j);
                newx[other_idx] += xnodal[this_idx + block_offsets[i]];
                newy[other_idx] += ynodal[this_idx + block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hase = 1;
            // duplicate += (b->beast.n - (bi->points[b->n2 - 1] != ~0u) - (bi->points[b->n2 * b->n1 - 1] != ~0u));
            bi->neighboring_block_idx.east = iother;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target - blocks)) < i)
        {
            for (unsigned j = 0; j < b->bwest.n; ++j)
            {
                unsigned other_idx = info[iother].points[point_boundary_index_flipped(b->bwest.block.target, (b->bwest.block.target_id), j)];
                unsigned this_idx = point_boundary_index(b, BOUNDARY_ID_WEST, j);
                newx[other_idx] += xnodal[this_idx + block_offsets[i]];
                newy[other_idx] += ynodal[this_idx + block_offsets[i]];
                bi->points[this_idx] = other_idx;
                division_factor[other_idx] += 1;
            }
            hasw = 1;
            // duplicate += (b->beast.n - (bi->points[0] != ~0u) - (bi->points[b->n2 * (b->n1 - 1)] != ~0u));
            bi->neighboring_block_idx.west = iother;
        }
        // unsigned new_pts = b->n1 * b->n2 - duplicate;
        // unsigned offset = unique_pts;
        for (unsigned row = hass; row < b->n1 - hasn; ++row)
        {
            for (unsigned col = hasw; col < b->n2-hase; ++col)
            {
                geo_id idx = col + row * b->n2;
                assert(bi->points[idx] == ~0);
                bi->points[idx] = unique_pts;
                newx[unique_pts] = xnodal[block_offsets[i] + idx];
                newy[unique_pts] = ynodal[block_offsets[i] + idx];
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
    free(division_factor);

    free(xnodal);
    xnodal = newx;
    free(ynodal);
    ynodal = newy;
    curve* line_array = calloc(max_lines, sizeof(*line_array));
    assert(line_array);

    // Create mesh line info
    unsigned line_count = 1;
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        block_info* bi = info + i;
        const mesh_block* b = blocks + i;
        unsigned hasn = 0, hass = 0, hase = 0, hasw = 0;
        unsigned iother;
        bi->first_ln = line_count;
        if (b->bnorth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bnorth.block.target - blocks)) < i)
        {
            deal_with_line_boundary(&b->bnorth.block, info + i, info + iother);
            hasn = 1;
        }
        if (b->bsouth.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bsouth.block.target - blocks)) < i)
        {
            deal_with_line_boundary(&b->bsouth.block, info + i, info + iother);
            hass = 1;
        }
        if (b->beast.type == BOUNDARY_TYPE_BLOCK && (iother = (b->beast.block.target - blocks)) < i)
        {
            deal_with_line_boundary(&b->beast.block, info + i, info + iother);
            hase = 1;
        }
        if (b->bwest.type == BOUNDARY_TYPE_BLOCK && (iother = (b->bwest.block.target - blocks)) < i)
        {
            deal_with_line_boundary(&b->bwest.block, info + i, info + iother);
            hasw = 1;
        }

        for (unsigned row = hass; row < b->n1 - hasn; ++row)
        {
            for (unsigned col = 0; col < b->n2 - 1; ++col)
            {
                unsigned idx = row * b->n2 + col;
                unsigned n1 = bi->points[idx];
                unsigned n2 = bi->points[idx + 1];
                bi->lines[row * (b->n2 - 1) + col] = line_count;
                line_array[line_count - 1] = (curve){.pt1 = n1, .pt2 = n2};
                line_count += 1;
            }
        }

        for (unsigned col = hasw; col < b->n2 - hase; ++col)
        {
            for (unsigned row = 0; row < b->n1 - 1; ++row)
            {
                unsigned idx = row * b->n2 + col;
                unsigned n1 = bi->points[idx];
                unsigned n2 = bi->points[idx + b->n2];
                bi->lines[(b->n2 - 1) * b->n1 + col * (b->n1 - 1) + row] = line_count;
                line_array[line_count - 1] = (curve){.pt1 = n1, .pt2 = n2};
                line_count += 1;
            }
        }
        for (unsigned j = 0; j < (b->n2 - 1) * b->n1 + b->n2 * (b->n1 - 1); ++j)
        {
            assert(bi->lines[i] != 0);
        }
        bi->last_ln = line_count;
    }
    line_count -= 1;

    unsigned surf_count = 0;
    surface* surfaces = calloc(max_surfaces, sizeof*surfaces);
    assert(surfaces);
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const mesh_block* b = blocks + i;
        block_info* bi = info + i;
        for (unsigned row = 0; row < b->n1 - 1; ++row)
        {
            for (unsigned col = 0; col < b->n2 - 1; ++col)
            {
                geo_id btm = (geo_id)(bi->lines[col + row * (b->n2 - 1)]);
                geo_id top = (geo_id)(bi->lines[col + (row + 1) * (b->n2 - 1)]);
                geo_id lft = (geo_id)(bi->lines[b->n1 * (b->n2 - 1) + row + col * (b->n1 - 1)]);
                geo_id rgt = (geo_id)(bi->lines[b->n1 * (b->n2 - 1) + row + (col + 1) * (b->n1 - 1)]);
                assert(surf_count < max_surfaces);
                bi->surfaces[col + row*(b->n2 - 1)] = (geo_id)surf_count;
                surfaces[surf_count] = (surface){.cs = +btm, .ce = +rgt, .cn = -top, .cw = -lft};
                surf_count += 1;
            }
        }
    }
    assert(surf_count == max_surfaces);




    if (ret == MESH_SUCCESS)
    {
        p_out->n_blocks = n_blocks;
        p_out->p_x = xnodal;
        p_out->p_y = ynodal;
        p_out->block_info = info;
        p_out->n_points = unique_pts;
        p_out->n_curves = line_count;
        p_out->p_curves = line_array;
        p_out->n_surfaces = max_surfaces;
        p_out->p_surfaces = surfaces;
        surfaces = NULL;
        line_array = NULL;
        info = NULL;
        xnodal = NULL;
        ynodal = NULL;
    }
    if (info)
    {
        for (unsigned i = 0; i < n_blocks; ++i)
        {
            free_block_info(info + i);
        }
        free(info);
    }
    free(ynodal);
    free(xnodal);
    free(yrhs);
    free(xrhs);
    free(block_offsets);


    return ret;
}

void mesh_destroy(mesh* mesh)
{
    free(mesh->p_x);
    free(mesh->p_y);
    free(mesh->p_curves);
    free(mesh->p_surfaces);
    for (unsigned i = 0; i < mesh->n_blocks; ++i)
    {
        free_block_info(mesh->block_info + i);
    }
    free(mesh->block_info);
}
