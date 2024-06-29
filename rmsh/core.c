//
// Created by jan on 22.6.2024.
//
#define PY_SSIZE_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _rmshAPI
#include <numpy/npy_no_deprecated_api.h>
#include <numpy/ndarrayobject.h>

#include "../geometry.h"
#include "../mesh2d.h"
#include "../io.h"

static PyObject* rmsh_info_func(PyObject* self, PyObject* args)
{
    return PyUnicode_FromString("rmsh - dev version");
}

//  Mesh type Python interface
typedef struct PyMesh2dObject
{
    PyObject_HEAD
    mesh2d data;
} PyMesh2dObject;

static PyObject* mesh_getx(PyObject* self, void* unused)
{
    (void)unused;
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (this->data.p_x == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = this->data.n_points;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, this->data.p_x);
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
    return (PyObject*)arr;
}

static PyObject* mesh_gety(PyObject* self, void* unused)
{
    (void)unused;
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (this->data.p_y == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = this->data.n_points;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, this->data.p_y);
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
    return (PyObject*)arr;
}

static PyObject* mesh_getl(PyObject* self, void* unused)
{
    (void)unused;
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (this->data.p_lines == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = 2 * this->data.n_lines;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, NPY_INT32, this->data.p_lines);
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
    return (PyObject*)arr;
}

static PyObject* mesh_gets(PyObject* self, void* unused)
{
    (void)unused;
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (this->data.p_surfaces == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = 4 * this->data.n_surfaces;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, NPY_INT32, this->data.p_surfaces);
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
    return (PyObject*)arr;
}

static PyObject* mesh_block_lines(PyObject* self, PyObject* v)
{
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    const long idx = PyLong_AsLong(v);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (idx >= this->data.n_blocks || idx < 0)
    {
        return PyErr_Format(PyExc_IndexError, "Block index %l was out of bounds [0, %u)", idx, this->data.n_blocks);
    }
    if (this->data.block_info == NULL)
    {
        Py_RETURN_NONE;
    }
    npy_intp dims = (this->data.block_info[idx].n1 - 1) * this->data.block_info[idx].n2 +
        (this->data.block_info[idx].n2 - 1) * this->data.block_info[idx].n1;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, &dims, NPY_INT32, this->data.block_info[idx].lines);
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
    return (PyObject*)arr;
}

static PyObject* mesh_boundary_lines(PyObject* self, PyObject* v)
{
    unsigned block_idx, bndid;
    if (!PyArg_ParseTuple(v, "II", &block_idx, &bndid))
    {
        return NULL;
    }
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (block_idx >= this->data.n_blocks)
    {
        return PyErr_Format(PyExc_IndexError, "Invalid index of %u was passed when the mesh only has %u blocks", block_idx, this->data.n_blocks);
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

    const geo_id* first;
    unsigned cnt;
    int stride;
    error_id res = mesh2d_get_boundary_lines_info(&this->data, block_idx, id, &first, &cnt, &stride);
    if (res != MESH_SUCCESS)
    {
        return PyErr_Format(PyExc_RuntimeError, "Failed to retrieve block line info (error code %d)", res);
    }
    npy_intp dims = cnt;
    npy_intp strides = sizeof(*first)*stride;

    PyArrayObject* arr = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, &dims, NPY_INT32, &strides, (void*)first, sizeof(*first), 0, NULL);
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
    return (PyObject*)arr;

}

static PyObject* mesh_boundary_points(PyObject* self, PyObject* v)
{
    unsigned block_idx, bndid;
    if (!PyArg_ParseTuple(v, "II", &block_idx, &bndid))
    {
        return NULL;
    }
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    if (block_idx >= this->data.n_blocks)
    {
        return PyErr_Format(PyExc_IndexError, "Invalid index of %u was passed when the mesh only has %u blocks", block_idx, this->data.n_blocks);
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

    const geo_id* first;
    unsigned cnt;
    int stride;
    error_id res = mesh2d_get_boundary_points_info(&this->data, block_idx, id, &first, &cnt, &stride);
    if (res != MESH_SUCCESS)
    {
        return PyErr_Format(PyExc_RuntimeError, "Failed to retrieve block line info (error code %d)", res);
    }
    npy_intp dims = cnt;
    npy_intp strides = sizeof(*first)*stride;
    PyArrayObject* arr = (PyArrayObject*)PyArray_New(&PyArray_Type, 1, &dims, NPY_INT32, &strides, (void*)first, sizeof(*first), 0, NULL);
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
    return (PyObject*)arr;

}

static PyGetSetDef mesh_getset[] =
    {
        {"x", mesh_getx, NULL, "X coordinates of nodes", NULL},
        {"y", mesh_gety, NULL, "Y coordinates of nodes", NULL},
        {"l", mesh_getl, NULL, "line indices", NULL},
        {"s", mesh_gets, NULL, "surface indices", NULL},
        {NULL} //  Sentinel
    };

static PyMethodDef mesh_methods[] =
    {
        {.ml_name = "blines", .ml_meth = mesh_block_lines, .ml_flags = METH_O, .ml_doc = "retrieves indices of lines which are in a block with the given index"},
        {.ml_name = "boundary_lines", .ml_meth = mesh_boundary_lines, .ml_flags = METH_VARARGS, .ml_doc = "Retrieves line indices for a specified boundary of a block"},
        {.ml_name = "boundary_pts", .ml_meth = mesh_boundary_points, .ml_flags = METH_VARARGS, .ml_doc = "Retrieves point indices for a specified boundary of a block"},
        {NULL}  //  Sentinel
    };

static void* wrap_alloc(void* state, size_t sz)
{
    (void)state;
    return PyMem_MALLOC(sz);
}

static void* wrap_realloc(void* state, void* ptr, size_t newsz)
{
    (void)state;
    return PyMem_REALLOC(ptr, newsz);
}

static void wrap_free(void* state, void* ptr)
{
    (void)state;
    PyMem_FREE(ptr);
}


static void mesh_dtor(PyObject* self)
{
    allocator a =
        {
        .alloc = wrap_alloc,
        .realloc = wrap_realloc,
        .free = wrap_free
    };
    PyMesh2dObject* this = (PyMesh2dObject*)self;
    mesh_destroy(&this->data, &a);
}

static PyTypeObject mesh_type =
    {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "_rmsh.mesh",
        .tp_doc = "internal mesh interface",
        .tp_basicsize = sizeof(PyMesh2dObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT|Py_TPFLAGS_IMMUTABLETYPE|Py_TPFLAGS_DISALLOW_INSTANTIATION,
        .tp_new = PyType_GenericNew,
        .tp_getset = mesh_getset,
        .tp_finalize = mesh_dtor,
        .tp_methods = mesh_methods,
    };




//  Mesh creation function
static PyObject* rmsh_create_mesh_function(PyObject* self, PyObject* args)
{
    PyListObject* input_list = NULL;
    PyTupleObject* options = NULL;
    int b_verbose = 0;
    if (!PyArg_ParseTuple(args, "O!pO!", &PyList_Type, &input_list, &b_verbose, &PyTuple_Type, &options))
    {
        //  Failed parsing the input for some reason, return NULL
        return NULL;
    }

    //  Unpack solver options
    solver_config cfg;
    if (!PyArg_ParseTuple((PyObject*)options, "pdIII", &cfg.direct, &cfg.tol, &cfg.smoother_rounds, &cfg.max_iterations, &cfg.max_rounds))
    {
        return NULL;
    }
    cfg.verbose = b_verbose;
    if (b_verbose) printf("Parsed the input args\n");



    //  Convert to usable form
    const unsigned n_blocks = PyList_GET_SIZE(input_list);
    mesh2d_block* p_blocks = PyMem_MALLOC(n_blocks * sizeof*p_blocks);
    if (p_blocks == NULL)
    {
        return PyErr_NoMemory();
    }
    if (b_verbose) printf("Allocated memory for the blocks\n");
    for (unsigned i = 0; i < n_blocks; ++i)
    {
        const char* label = NULL;
        PyTupleObject* bnds[4];
        if (!PyArg_ParseTuple(PyList_GET_ITEM(input_list, i), "sO!O!O!O!", &label, &PyTuple_Type, &bnds+0, &PyTuple_Type, bnds+1,
                              &PyTuple_Type, bnds+2, &PyTuple_Type, bnds+3))
        {
            PyMem_FREE(p_blocks);
            return NULL;
        }
        if (b_verbose) printf("Parsed the data for block \"%s\"\n", label);
        
        static const boundary_id ids[4] = {BOUNDARY_ID_NORTH, BOUNDARY_ID_SOUTH, BOUNDARY_ID_EAST, BOUNDARY_ID_WEST};
        for (unsigned j = 0; j < 4; ++j)
        {
            if (b_verbose) printf("\tDealing with the boundary \"%u\"\n", j);
            const boundary_id id = ids[j];
            PyTupleObject* t = bnds[j];
            boundary bnd;
            int bnd_id = 0;
            unsigned bnd_n = 0;
            if (PyTuple_GET_SIZE(t) != 5)
            {
                PyMem_FREE(p_blocks);
                return PyErr_Format(PyExc_RuntimeError, "Boundary tuple had an invalid length (should be 5)");
            }
            int bnd_type = (int)PyLong_AsLong(PyTuple_GET_ITEM(t, 0));
            if (b_verbose) printf("\t\tBoundary type: \"%d\"\n", bnd_type);
            if (bnd_type == 0)
            {
                //  Boundary Curve
                boundary_curve c;
                PyArrayObject* x,* y;
                if (!PyArg_ParseTuple((PyObject*)t, "iiIO!O!", &bnd_type, &bnd_id, &bnd_n, &PyArray_Type, &x, &PyArray_Type, &y))
                {
                    PyMem_FREE(p_blocks);
                    return NULL;
                }
                if (b_verbose) printf("\t\tParsed curve boundary\n");
                c.x = (double*)PyArray_DATA(x);
                c.y = (double*)PyArray_DATA(y);
                c.n = bnd_n;
                bnd = (boundary){.type = BOUNDARY_TYPE_CURVE, .curve= c};
            }
            else if (bnd_type == 1)
            {
                //  Boundary Block
                boundary_block b;
                unsigned target, target_id;
                if (!PyArg_ParseTuple((PyObject*)t, "iiIII", &bnd_type, &bnd_id, &bnd_n, &target, &target_id))
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
                if (b_verbose) printf("\t\tParsed block boundary\n");
                bnd = (boundary){.type = BOUNDARY_TYPE_BLOCK, .block=b};
            }
            else
            {
                PyMem_FREE(p_blocks);
                return PyErr_Format(PyExc_RuntimeError, "Boundary tuple type had an invalid value (should be 0 or 1)");
            }
            if (b_verbose) printf("\t\tWriting the boundary to the block\n");

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

    if (b_verbose) printf("Finished converting the inputs\n");
    //  Blocks should be all set up now

    PyMesh2dObject* msh = PyObject_New(PyMesh2dObject, &mesh_type);
    if (msh == NULL)
    {
        PyMem_FREE(p_blocks);
        return NULL;
    }
    if (b_verbose) printf("Calling the mesh function\n");
    allocator a =
        {
            .alloc = wrap_alloc,
            .realloc = wrap_realloc,
            .free = wrap_free
        };
    double rx, ry;
    const error_id e = mesh2d_create_elliptical(n_blocks, p_blocks, &cfg, &a, &msh->data, &rx, &ry);
    PyMem_FREE(p_blocks);
    if (e != MESH_SUCCESS)
    {
        Py_DECREF(msh);
        return PyErr_Format(PyExc_RuntimeError, "Failed mesh creation (error code %u)", (unsigned)e);
    }
    PyObject* ox = PyFloat_FromDouble(rx);
    if (!ox)
    {
        Py_DECREF(msh);
        return NULL;
    }
    PyObject* oy = PyFloat_FromDouble(ry);
    if (!oy)
    {
        Py_DECREF(ox);
        Py_DECREF(msh);
        return NULL;
    }
    PyObject* tout = PyTuple_Pack(3, msh, ox, oy);
    if (!tout)
    {
        Py_DECREF(oy);
        Py_DECREF(ox);
        Py_DECREF(msh);
        return NULL;
    }

    if (b_verbose) printf("Returning from the function\n");
    return tout;
}




static PyMethodDef module_methods[] =
    {
        {.ml_name = "info", .ml_meth = rmsh_info_func, .ml_flags = METH_NOARGS, .ml_doc = "Prints the info about th module"},
        {.ml_name = "create_elliptical_mesh", .ml_meth = rmsh_create_mesh_function, .ml_flags = METH_VARARGS, .ml_doc = "Internal module function, which creates an elliptical mesh"},
        //  Terminating entry
        {NULL, NULL, 0, NULL},
    };

static PyModuleDef module_definition =
    {
        .m_base= PyModuleDef_HEAD_INIT,
        .m_name = "_rmsh",
        .m_doc = "Rectangular Mesh Generator",
        .m_size = -1,
        .m_methods = module_methods,
    };

PyMODINIT_FUNC PyInit__rmsh(void)
{
    import_array();
    if (PyArray_ImportNumPyAPI() < 0) {
        return NULL;
    }
    if (PyType_Ready(&mesh_type) < 0)
    {
        return NULL;
    }
    PyObject* m = PyModule_Create(&module_definition);
    if (m == NULL)
    {
        return m;
    }
    Py_INCREF(&mesh_type);
    if (PyModule_AddObject(m, "mesh", (PyObject*)&mesh_type) < 0)
    {
        Py_DECREF(&mesh_type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
