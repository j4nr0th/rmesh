//
// Created by jan on 22.6.2024.
//

#define PY_SSIZE_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _rmshAPI
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/npy_no_deprecated_api.h>

#include "err.h"
#include "geometry.h"
#include "mesh2d.h"

//  Mesh type Python interface
typedef struct PyMesh2dObject
{
    PyObject_HEAD mesh2d data;
} PyMesh2dObject;

static PyObject *mesh_getx(PyObject *self, void *unused)
{
    (void)unused;
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (this->data.p_x == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = this->data.n_points;

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, this->data.p_x);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_gety(PyObject *self, void *unused)
{
    (void)unused;
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (this->data.p_y == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = this->data.n_points;

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, this->data.p_y);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_getl(PyObject *self, void *unused)
{
    (void)unused;
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (this->data.p_lines == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims[2] = {this->data.n_lines, 2};

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT, this->data.p_lines);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_gets(PyObject *self, void *unused)
{
    (void)unused;
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (this->data.p_surfaces == NULL)
    {
        Py_RETURN_NONE;
    }
    const npy_intp dims[2] = {this->data.n_surfaces, 4};

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_INT32, this->data.p_surfaces);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const PyMesh2dObject *this = (PyMesh2dObject *)self;
    return PyLong_FromSize_t((size_t)this->data.n_points);
}

static PyObject *mesh_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const PyMesh2dObject *this = (PyMesh2dObject *)self;
    return PyLong_FromSize_t((size_t)this->data.n_lines);
}

static PyObject *mesh_get_n_surfaces(PyObject *self, void *Py_UNUSED(closure))
{
    const PyMesh2dObject *this = (PyMesh2dObject *)self;
    return PyLong_FromSize_t((size_t)this->data.n_surfaces);
}

static PyObject *mesh_block_lines(PyObject *self, PyObject *v)
{
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    const long idx = PyLong_AsLong(v);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (idx >= this->data.n_blocks || idx < 0)
    {
        return PyErr_Format(PyExc_IndexError, "Block index %li was out of bounds [0, %u)", idx, this->data.n_blocks);
    }
    if (this->data.block_info == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = (this->data.block_info[idx].n1 - 1) * this->data.block_info[idx].n2 +
                    (this->data.block_info[idx].n2 - 1) * this->data.block_info[idx].n1;

    PyArrayObject *arr =
        (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims, NPY_INT32, this->data.block_info[idx].lines);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_boundary_lines(PyObject *self, PyObject *v)
{
    unsigned block_idx, bndid;
    if (!PyArg_ParseTuple(v, "II", &block_idx, &bndid))
    {
        return NULL;
    }
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (block_idx >= this->data.n_blocks)
    {
        return PyErr_Format(PyExc_IndexError, "Invalid index of %u was passed when the mesh only has %u blocks",
                            block_idx, this->data.n_blocks);
    }
    boundary_id id;
    switch ((boundary_id)bndid)
    {
    case BOUNDARY_ID_EAST:
    case BOUNDARY_ID_WEST:
    case BOUNDARY_ID_NORTH:
    case BOUNDARY_ID_SOUTH:
        id = bndid;
        break;
    default:
        return PyErr_Format(PyExc_ValueError, "Boundary ID has an invalid value of %u", bndid);
    }

    const geo_id *first;
    unsigned cnt;
    int stride;
    error_id res = mesh2d_get_boundary_lines_info(&this->data, block_idx, id, &first, &cnt, &stride);
    if (res != MESH_SUCCESS)
    {
        return PyErr_Format(PyExc_RuntimeError, "Failed to retrieve block line info (%s: %s)", error_id_to_str(res),
                            error_id_to_msg(res));
    }
    npy_intp dims = cnt;
    npy_intp strides = sizeof(*first) * stride;

    PyArrayObject *arr = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &dims, NPY_INT32, &strides, (void *)first,
                                                      sizeof(*first), 0, NULL);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_boundary_points(PyObject *self, PyObject *v)
{
    unsigned block_idx, bndid;
    if (!PyArg_ParseTuple(v, "II", &block_idx, &bndid))
    {
        return NULL;
    }
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (block_idx >= this->data.n_blocks)
    {
        return PyErr_Format(PyExc_IndexError, "Invalid index of %u was passed when the mesh only has %u blocks",
                            block_idx, this->data.n_blocks);
    }
    boundary_id id;
    switch ((boundary_id)bndid)
    {
    case BOUNDARY_ID_EAST:
    case BOUNDARY_ID_WEST:
    case BOUNDARY_ID_NORTH:
    case BOUNDARY_ID_SOUTH:
        id = bndid;
        break;
    default:
        return PyErr_Format(PyExc_ValueError, "Boundary ID has an invalid value of %u", bndid);
    }

    const geo_id *first;
    unsigned cnt;
    int stride;
    error_id res = mesh2d_get_boundary_points_info(&this->data, block_idx, id, &first, &cnt, &stride);
    if (res != MESH_SUCCESS)
    {
        return PyErr_Format(PyExc_RuntimeError, "Failed to retrieve block line info (%s: %s)", error_id_to_str(res),
                            error_id_to_msg(res));
    }
    npy_intp dims = cnt;
    npy_intp strides = sizeof(*first) * stride;
    PyArrayObject *arr = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &dims, NPY_INT32, &strides, (void *)first,
                                                      sizeof(*first), 0, NULL);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_boundary_surfaces(PyObject *self, PyObject *v)
{
    unsigned block_idx, bndid;
    if (!PyArg_ParseTuple(v, "II", &block_idx, &bndid))
    {
        return NULL;
    }
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    if (block_idx >= this->data.n_blocks)
    {
        return PyErr_Format(PyExc_IndexError, "Invalid index of %u was passed when the mesh only has %u blocks",
                            block_idx, this->data.n_blocks);
    }
    boundary_id id;
    switch ((boundary_id)bndid)
    {
    case BOUNDARY_ID_EAST:
    case BOUNDARY_ID_WEST:
    case BOUNDARY_ID_NORTH:
    case BOUNDARY_ID_SOUTH:
        id = bndid;
        break;
    default:
        return PyErr_Format(PyExc_ValueError, "Boundary ID has an invalid value of %u", bndid);
    }

    const geo_id *first;
    unsigned cnt;
    int stride;
    error_id res = mesh2d_get_boundary_surface_info(&this->data, block_idx, id, &first, &cnt, &stride);
    if (res != MESH_SUCCESS)
    {
        return PyErr_Format(PyExc_RuntimeError, "Failed to retrieve block line info (%s: %s)", error_id_to_str(res),
                            error_id_to_msg(res));
    }
    npy_intp dims = cnt;
    npy_intp strides = sizeof(*first) * stride;
    PyArrayObject *arr = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &dims, NPY_INT32, &strides, (void *)first,
                                                      sizeof(*first), 0, NULL);
    if (arr)
    {
        Py_INCREF(self);
        if (PyArray_SetBaseObject(arr, self) != 0)
        {
            Py_DECREF(self);
            Py_DECREF(arr);
            arr = NULL;
        }
    }
    return (PyObject *)arr;
}

static PyObject *mesh_surface_element(PyObject *self, PyObject *v)
{
    int surface_id;
    unsigned order;
    if (!PyArg_ParseTuple(v, "iI", &surface_id, &order))
    {
        return NULL;
    }
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    npy_intp dims[2] = {(2 * order + 1), (2 * order + 1)};

    PyObject *arr = PyArray_New(&PyArray_Type, 2, dims, NPY_INT32, NULL, NULL, 0, 0, NULL);
    if (!arr)
    {
        return NULL;
    }
    static_assert(sizeof(npy_int32) == sizeof(geo_id));
    geo_id *out_array = PyArray_DATA((PyArrayObject *)arr);
    const error_id res = surface_centered_element(&this->data, (geo_id)surface_id, order, out_array);
    if (res != MESH_SUCCESS)
    {
        Py_DECREF(arr);
        return PyErr_Format(PyExc_RuntimeError,
                            "Could not create the surface element of order %u centered on index %d (%s: %s)", order,
                            surface_id, error_id_to_str(res), error_id_to_msg(res));
    }
    return arr;
}

static PyObject *mesh_surface_element_points(PyObject *self, PyObject *v)
{
    int surface_id;
    unsigned order;
    if (!PyArg_ParseTuple(v, "iI", &surface_id, &order))
    {
        return NULL;
    }
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    npy_intp dims[2] = {(2 * order + 2), (2 * order + 2)};

    PyObject *arr = PyArray_New(&PyArray_Type, 2, dims, NPY_INT32, NULL, NULL, 0, 0, NULL);
    if (!arr)
    {
        return NULL;
    }
    static_assert(sizeof(npy_int32) == sizeof(geo_id));
    geo_id *out_array = PyArray_DATA((PyArrayObject *)arr);
    const error_id res = surface_centered_element_points(&this->data, (geo_id)surface_id, order, out_array);
    if (res != MESH_SUCCESS)
    {
        Py_DECREF(arr);
        return PyErr_Format(PyExc_RuntimeError,
                            "Could not create the surface element of order %u centered on index %d (%s %s)", order,
                            surface_id, error_id_to_str(res), error_id_to_msg(res));
    }
    return arr;
}

static PyGetSetDef mesh_getset[] = {
    {"pos_x", mesh_getx, NULL, "NDArray[np.double] : X coordinates of nodes", NULL},
    {"pos_y", mesh_gety, NULL, "NDArray[np.double] : Y coordinates of nodes", NULL},
    {"lines", mesh_getl, NULL,
     "NDArray[np.int32] : line indices\n"
     "\n"
     "Has a shape ``(N, 2)``, where ``N`` is the number of lines in the mesh.\n",
     NULL},
    {"surfaces", mesh_gets, NULL,
     "NDArray[np.int32] : surface indices\n"
     "\n"
     "Has a shape ``(N, 4)``, where ``N`` is the number of surfaces in the mesh.\n"
     "Indices start at 1 instead of 0 and a negative value of the index means\n"
     "that a line should be in opposite orientation to how it is in the ``lines``\n"
     "array to maintain a consistent surface orientation.\n",
     NULL},
    {"n_points", mesh_get_n_points, NULL, "int : Number of points in the mesh.\n", NULL},
    {"n_lines", mesh_get_n_lines, NULL, "int : Number of lines in the mesh.\n", NULL},
    {"n_points", mesh_get_n_surfaces, NULL, "int : Number of surfaces in the mesh.\n", NULL},
    {NULL} //  Sentinel
};

static void *wrap_alloc(void *state, size_t sz)
{
    (void)state;
    return PyMem_Malloc(sz);
}

static void *wrap_realloc(void *state, void *ptr, size_t newsz)
{
    (void)state;
    return PyMem_Realloc(ptr, newsz);
}

static void wrap_free(void *state, void *ptr)
{
    (void)state;
    PyMem_Free(ptr);
}

//  Mesh creation function
static PyObject *rmsh_create_mesh_function(PyObject *type, PyObject *args)
{
    PyTypeObject *mesh_type = (PyTypeObject *)type;
    if (!mesh_type)
    {
        return NULL;
    }
    PyObject *input_list = NULL;
    PyObject *options = NULL;
    int b_verbose = 0;
    if (!PyArg_ParseTuple(args, "O!pO!", &PyList_Type, &input_list, &b_verbose, &PyTuple_Type, &options))
    {
        //  Failed parsing the input for some reason, return NULL
        return NULL;
    }

    //  Unpack solver options
    solver_config cfg;
    if (!PyArg_ParseTuple((PyObject *)options, "pdIII", &cfg.direct, &cfg.tol, &cfg.smoother_rounds,
                          &cfg.max_iterations, &cfg.max_rounds))
    {
        return NULL;
    }
    cfg.verbose = b_verbose;
    if (b_verbose)
        printf("Parsed the input args\n");

    //  Convert to usable form
    const unsigned n_blocks = PyList_Size(input_list);
    mesh2d_block *p_blocks = PyMem_MALLOC(n_blocks * sizeof *p_blocks);
    if (p_blocks == NULL)
    {
        return PyErr_NoMemory();
    }
    if (b_verbose)
        printf("Allocated memory for the blocks\n");
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const char *label = NULL;
        PyObject *bnds[4];
        if (!PyArg_ParseTuple(PyList_GetItem(input_list, i), "sO!O!O!O!", &label, &PyTuple_Type, &bnds + 0,
                              &PyTuple_Type, bnds + 1, &PyTuple_Type, bnds + 2, &PyTuple_Type, bnds + 3))
        {
            PyMem_FREE(p_blocks);
            return NULL;
        }
        if (b_verbose)
            printf("Parsed the data for block \"%s\"\n", label);

        static const boundary_id ids[4] = {BOUNDARY_ID_NORTH, BOUNDARY_ID_SOUTH, BOUNDARY_ID_EAST, BOUNDARY_ID_WEST};
        for (unsigned j = 0; j < 4; ++j)
        {
            if (b_verbose)
                printf("\tDealing with the boundary \"%u\"\n", j);
            const boundary_id id = ids[j];
            PyObject *t = bnds[j];
            boundary bnd;
            int bnd_id = 0;
            unsigned bnd_n = 0;
            if (PyTuple_Size(t) != 5)
            {
                PyMem_FREE(p_blocks);
                return PyErr_Format(PyExc_RuntimeError, "Boundary tuple had an invalid length (should be 5)");
            }
            int bnd_type = (int)PyLong_AsLong(PyTuple_GetItem(t, 0));
            if (b_verbose)
                printf("\t\tBoundary type: \"%d\"\n", bnd_type);
            if (bnd_type == 0)
            {
                //  Boundary Curve
                boundary_curve c;
                PyArrayObject *x, *y;
                if (!PyArg_ParseTuple((PyObject *)t, "iiIO!O!", &bnd_type, &bnd_id, &bnd_n, &PyArray_Type, &x,
                                      &PyArray_Type, &y))
                {
                    PyMem_FREE(p_blocks);
                    return NULL;
                }
                if (b_verbose)
                    printf("\t\tParsed curve boundary\n");
                c.x = (double *)PyArray_DATA(x);
                c.y = (double *)PyArray_DATA(y);
                c.n = bnd_n;
                bnd = (boundary){.type = BOUNDARY_TYPE_CURVE, .curve = c};
            }
            else if (bnd_type == 1)
            {
                //  Boundary Block
                boundary_block b;
                unsigned target, target_id;
                if (!PyArg_ParseTuple((PyObject *)t, "iiIII", &bnd_type, &bnd_id, &bnd_n, &target, &target_id))
                {
                    PyMem_FREE(p_blocks);
                    return NULL;
                }
                b.n = bnd_n;
                b.owner = i;
                b.owner_id = id;
                b.target = target;
                switch (target_id)
                {
                case BOUNDARY_ID_NORTH:
                    b.target_id = BOUNDARY_ID_NORTH;
                    break;
                case BOUNDARY_ID_SOUTH:
                    b.target_id = BOUNDARY_ID_SOUTH;
                    break;
                case BOUNDARY_ID_EAST:
                    b.target_id = BOUNDARY_ID_EAST;
                    break;
                case BOUNDARY_ID_WEST:
                    b.target_id = BOUNDARY_ID_WEST;
                    break;
                default:
                    PyMem_FREE(p_blocks);
                    return PyErr_Format(PyExc_RuntimeError, "Invalid block boundary type %u", target_id);
                }
                if (b_verbose)
                    printf("\t\tParsed block boundary\n");
                bnd = (boundary){.type = BOUNDARY_TYPE_BLOCK, .block = b};
            }
            else
            {
                PyMem_FREE(p_blocks);
                return PyErr_Format(PyExc_RuntimeError, "Boundary tuple type had an invalid value (should be 0 or 1)");
            }
            if (b_verbose)
                printf("\t\tWriting the boundary to the block\n");

            switch (id)
            {
            case BOUNDARY_ID_NORTH:
                p_blocks[i].bnorth = bnd;
                break;
            case BOUNDARY_ID_SOUTH:
                p_blocks[i].bsouth = bnd;
                break;
            case BOUNDARY_ID_EAST:
                p_blocks[i].beast = bnd;
                break;
            case BOUNDARY_ID_WEST:
                p_blocks[i].bwest = bnd;
                break;
            }
        }
    }

    if (b_verbose)
        printf("Finished converting the inputs\n");
    //  Blocks should be all set up now

    PyMesh2dObject *msh = (PyMesh2dObject *)mesh_type->tp_alloc(mesh_type, 0);
    if (msh == NULL)
    {
        PyMem_FREE(p_blocks);
        return NULL;
    }
    if (b_verbose)
        printf("Calling the mesh function\n");
    allocator a = {.alloc = wrap_alloc, .realloc = wrap_realloc, .free = wrap_free};
    double rx, ry;
    const error_id e = mesh2d_create_elliptical(n_blocks, p_blocks, &cfg, &a, &msh->data, &rx, &ry);
    PyMem_FREE(p_blocks);
    if (e != MESH_SUCCESS)
    {
        Py_DECREF(msh);
        return PyErr_Format(PyExc_RuntimeError, "Failed mesh creation (%s: %s)", error_id_to_str(e),
                            error_id_to_msg(e));
    }
    PyObject *ox = PyFloat_FromDouble(rx);
    if (!ox)
    {
        Py_DECREF(msh);
        return NULL;
    }
    PyObject *oy = PyFloat_FromDouble(ry);
    if (!oy)
    {
        Py_DECREF(ox);
        Py_DECREF(msh);
        return NULL;
    }
    PyObject *tout = PyTuple_Pack(3, msh, ox, oy);
    if (!tout)
    {
        Py_DECREF(oy);
        Py_DECREF(ox);
        Py_DECREF(msh);
        return NULL;
    }

    if (b_verbose)
        printf("Returning from the function\n");
    return tout;
}

static PyMethodDef mesh_methods[] = {
    {.ml_name = "block_lines",
     .ml_meth = mesh_block_lines,
     .ml_flags = METH_O,
     .ml_doc = "block_lines(idx: int, /) -> npt.NDArray[np.int32]\n"
               "Return indices of all lines within a block.\n"
               "\n"
               "Indices start at 1 and a negative value indicates a reversed orientation of\n"
               "the line.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "block_id : int\n"
               "    The index of the block for which the line indices should be returned.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all lines within the mesh block specified by\n"
               "    ``block_id``.\n"},
    {.ml_name = "block_boundary_lines",
     .ml_meth = mesh_boundary_lines,
     .ml_flags = METH_VARARGS,
     .ml_doc = "block_boundary_lines(block_id: int, boundary_id: int, /) -> npt.NDArray[np.int32]\n"
               "Return indices of all lines on a boundary of a block.\n"
               "\n"
               "Indices start at 1 and a negative value indicates a reversed orientation of the\n"
               "line.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "block_id : int\n"
               "    The index of the block for which the line indices should be returned.\n"
               "boundary : BoundaryId\n"
               "    The ID of a boundary from which the line indices should be returned.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all lines on a boundary of a block ``block_id``.\n"},
    {.ml_name = "block_boundary_points",
     .ml_meth = mesh_boundary_points,
     .ml_flags = METH_VARARGS,
     .ml_doc = "block_boundary_points(block_id: int, boundary: BoundaryId) -> npt.NDArray[np.int32]\n"
               "Return indices of all nodes on a boundary of a block.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "block_id : int\n"
               "    The index of the block for which the point indices should be returned.\n"
               "boundary : BoundaryId\n"
               "    The ID of a boundary from which the point indices should be returned.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all points on a boundary of a block ``block_id``.\n"},
    {.ml_name = "block_boundary_surfaces",
     .ml_meth = mesh_boundary_surfaces,
     .ml_flags = METH_VARARGS,
     .ml_doc = "block_boundary_surfaces(block_id: str, boundary: BoundaryId) -> npt.NDArray[np.int32]\n"
               "Return indices of all surfaces on a boundary of a block.\n"
               "\n"
               "Indices start at 1 and a negative value indicates a reversed orientation\n"
               "of the surface, though for this function this is not needed.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "block_id : str\n"
               "    The label of the block for which the surfaces indices should be returned.\n"
               "boundary : BoundaryId\n"
               "    The ID of a boundary from which the surfaces indices should be returned.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all surfaces on a boundary of a block ``block_id``.\n"},
    {.ml_name = "surface_element",
     .ml_meth = mesh_surface_element,
     .ml_flags = METH_VARARGS,
     .ml_doc = "surface_element(surf, order) -> npt.NDArray[np.int32]\n"
               "Return indices of surfaces, which form a square element of width (2*order+1).\n"
               "\n"
               "This is intended to be used for computing cell-based interpolations.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "surf : int\n"
               "    The one-based index of the surface which should be the center of the element.\n"
               "order : int\n"
               "    Size of the element in each direction away from the center (zero means only\n"
               "    the element, one means 3 x 3, etc.)\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all surfaces in the element. Note that since one-based\n"
               "    indexing is used, a zero indicates a missing surface caused by a numerical\n"
               "    boundary. Negative indices mean a negative orientation.\n"},
    {.ml_name = "surface_element_points",
     .ml_meth = mesh_surface_element_points,
     .ml_flags = METH_VARARGS,
     .ml_doc = "surface_element_points(surf: int, order: int) -> npt.NDArray[np.int32]\n"
               "Return indices of points, which form a square element of width (2*order+1).\n"
               "\n"
               "This is intended to be used for computing nodal-based interpolations for surface\n"
               "elements.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "surf : int\n"
               "    The one-based index of the surface which should be the center of the element.\n"
               "order : int\n"
               "    Size of the element in each direction away from the center (zero means only\n"
               "    the element, one means 3 x 3, etc.)\n"
               "\n"
               "Returns\n"
               "-------\n"
               "ndarray[int32]\n"
               "    Array with indices of all indices in the element. Note that since one-based\n"
               "    indexing is used, a value of -1 indicates a missing point caused by a\n"
               "    numerical boundary.\n"

    },
    {.ml_name = "_create_elliptical_mesh",
     .ml_meth = rmsh_create_mesh_function,
     .ml_flags = METH_CLASS | METH_VARARGS,
     .ml_doc = "_create_elliptical_mesh(\n"
               "    arg1: list[_BlockInfoTuple],\n"
               "    arg2: bool,\n"
               "    arg3: _SolverCfgTuple,\n"
               "    /,\n"
               ") -> tuple[Self, float, float]\n"
               "Create an elliptical mesh.\n"
               "\n"
               "This method takes in *heavily* pre-processed input. This is for the sake of\n"
               "making the parsing in C as simple as possible.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "arg1 : list of _BlockInfoTuple\n"
               "    List of tuples which contain information about mesh blocks.\n"
               "arg2 : bool\n"
               "    Verbosity setting.\n"
               "arg3 : _SolverCfgTuple\n"
               "    Tuple containing pre-processed solver config values.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    The newly created mesh object.\n"
               "float\n"
               "    Residual of the x-equation.\n"
               "float\n"
               "    Residual of the y-equation.\n"},
    {NULL} //  Sentinel
};

static void mesh_dtor(PyObject *self)
{
    allocator a = {.alloc = wrap_alloc, .realloc = wrap_realloc, .free = wrap_free};
    PyMesh2dObject *this = (PyMesh2dObject *)self;
    mesh_destroy(&this->data, &a);
}

static PyTypeObject mesh_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "rmsh._rmsh._Mesh2D",
    .tp_basicsize = sizeof(PyMesh2dObject),
    .tp_itemsize = 0,
    .tp_new = PyType_GenericNew,
    .tp_getset = mesh_getset,
    .tp_doc = "internal mesh interface",
    .tp_finalize = mesh_dtor,
    .tp_methods = mesh_methods,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION | Py_TPFLAGS_BASETYPE,
};

static PyModuleDef module_definition = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_rmsh",
    .m_doc = "Rectangular Mesh Generator",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__rmsh(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return NULL;
    }
    PyObject *m = PyModule_Create(&module_definition);
    if (m == NULL)
    {
        return m;
    }
    if (PyModule_AddType(m, &mesh_type) < 0)
    {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
