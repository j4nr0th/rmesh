//
// Created by jan on 15.6.2024.
//

#include "err.h"


#define MSG_TABLE_ENTRY(id, message) [id] = {.str = #id, .msg = message}

static const struct {const char* str; const char *msg;} message_table[MESH_COUNT] =
    {
        MSG_TABLE_ENTRY(MESH_SUCCESS, "Function returned successfully"),
        MSG_TABLE_ENTRY(MESH_BOUNDARY_SIZE_MISMATCH, "Opposing boundaries of a block don't have the same number of points"),
        MSG_TABLE_ENTRY(MESH_ALLOCATION_FAILED, "Memory allocation failed"),
        MSG_TABLE_ENTRY(MESH_MATRIX_FAILURE, "An error occurred during the construction of the system matrix"),
        MSG_TABLE_ENTRY(MESH_SOLVER_FAILED, "Solver was unable to find the locations of the nodes"),
        MSG_TABLE_ENTRY(MESH_INVALID_BOUNDARY_ID, "Boundary ID did not have a valid value"),
        MSG_TABLE_ENTRY(MESH_BOUNDARY_UNSORTED, "Two blocks share a curve/block boundary and the block boundary was defined on the first of the two"),
        MSG_TABLE_ENTRY(MESH_INDEX_OUT_OF_BOUNDS, "Index of a block was too large"),
    };

static const char* unknown_str = "Unknown";

const char* error_id_to_str(error_id id)
{
    if (id < 0 || id >= MESH_COUNT)
    {
        return unknown_str;
    }
    return message_table[id].str;
}

const char* error_id_to_msg(error_id id)
{
    if (id < 0 || id >= MESH_COUNT)
    {
        return unknown_str;
    }
    return message_table[id].msg;
}
