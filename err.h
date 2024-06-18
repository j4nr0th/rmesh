//
// Created by jan on 15.6.2024.
//

#ifndef MESH_ERR_H
#define MESH_ERR_H

typedef enum error_id_enum error_id;
enum error_id_enum
{
    MESH_SUCCESS,
    MESH_BOUNDARY_SIZE_MISMATCH,
    MESH_ALLOCATION_FAILED,
    MESH_MATRIX_FAILURE,
    MESH_SOLVER_FAILED,
};

#endif //MESH_ERR_H
