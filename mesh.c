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


error_id mesh_create(unsigned int n_blocks, mesh_block* blocks, mesh* p_out)
{
    error_id ret = MESH_SUCCESS;
    unsigned point_cnt = 0;
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
        ret = MESH_ALLOCATION_FAILED;
        goto mtx_allocation_failed;
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
        goto mtx_allocation_failed;
    }

    if (ret == MESH_SUCCESS)
    {
        p_out->n_points = point_cnt;
        p_out->p_x = xnodal;
        p_out->p_y = ynodal;
        xnodal = NULL;
        ynodal = NULL;
    }
mtx_allocation_failed:
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
}
