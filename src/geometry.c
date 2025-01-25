//
// Created by jan on 30.6.2024.
//
#include "geometry.h"

const char *boundary_type_to_str(boundary_type type)
{
    switch (type)
    {
    case BOUNDARY_TYPE_CURVE:
        return "Curve";
    case BOUNDARY_TYPE_BLOCK:
        return "Block";
    case BOUNDARY_TYPE_NONE:
        return "None";
    default:
        return "Unknown";
    }
}

const char *boundary_id_to_str(boundary_id id)
{
    switch (id)
    {
    case BOUNDARY_ID_EAST:
        return "East";
    case BOUNDARY_ID_NORTH:
        return "North";
    case BOUNDARY_ID_SOUTH:
        return "South";
    case BOUNDARY_ID_WEST:
        return "West";
    default:
        return "Unknown";
    }
}
