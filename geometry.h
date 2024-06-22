//
// Created by jan on 15.6.2024.
//

#ifndef MESH_GEOMETRY_H
#define MESH_GEOMETRY_H

typedef struct boundary_struct boundary;
typedef struct mesh_block_struct mesh2d_block;

typedef int geo_id;

enum boundary_type_enum
{
    BOUNDARY_TYPE_NONE = 0,
    BOUNDARY_TYPE_CURVE = 1,
    BOUNDARY_TYPE_BLOCK = 2,
};
typedef enum boundary_type_enum boundary_type;

enum boundary_id_enum
{
    BOUNDARY_ID_NORTH = 1,
    BOUNDARY_ID_EAST = 2,
    BOUNDARY_ID_WEST = 3,
    BOUNDARY_ID_SOUTH = 4,
};
typedef enum boundary_id_enum boundary_id;


struct boundary_curve_struct
{
    unsigned n;
    double* x;
    double* y;
};
typedef struct boundary_curve_struct boundary_curve;

struct boundary_block_struct
{
    unsigned n;
    geo_id owner;
    boundary_id owner_id;
    geo_id target;
    boundary_id target_id;
};

typedef struct boundary_block_struct boundary_block;

struct boundary_struct
{
    boundary_type type;
    union
    {
        unsigned n;
        boundary_curve curve;
        boundary_block block;
    };
};

struct mesh_block_struct
{
    const char* label;
    boundary bnorth;
    boundary beast;
    boundary bwest;
    boundary bsouth;
};

#endif //MESH_GEOMETRY_H
