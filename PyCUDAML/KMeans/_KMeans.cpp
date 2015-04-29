#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "KMeans.hpp"

/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for fitting KMeans clustering in C++.";
static char kmeans_docstring[] =
    "Fit KMeans clustering of the given data.";

/* Available functions */
static PyObject *KMeans_kmeans(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"kmeans", KMeans_kmeans, METH_VARARGS, kmeans_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_KMeans(void)
{
    PyObject *m = Py_InitModule3("_KMeans", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *KMeans_kmeans(PyObject *self, PyObject *args)
{
    int k, max_iter, threshold;
    PyObject *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iOii", &k, &X_obj, &max_iter, &threshold))
        return NULL;

    int typenum = NPY_FLOAT;

    /* Interpret the input objects as numpy arrays. */
    PyObject *X_array = PyArray_FROM_OTF(X_obj, typenum, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (X_array == NULL) {
        Py_XDECREF(X_array);
        return NULL;
    }

    /* How many data points are there? */
    int n = (int)PyArray_DIM(X_array, 0);
    int d = (int)PyArray_DIM(X_array, 1);

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    npy_intp dims[2];
    float **X;

    /* Get pointers to the data as C-types. */
    if (PyArray_AsCArray(&X_array, (void **)&X, dims, 2, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        Py_DECREF(descr);
        Py_DECREF(X_array);
        return NULL;
    }

    /* Run KMeans clustering */
    float **cluster_centers = NULL;
    if (!(cluster_centers = (float**)malloc(k*sizeof(float*))))
    {
        throw;
    }
    kmeans(k, (const float **) X, n, d, max_iter, threshold, cluster_centers);

    /* Build the output tuple */
    dims[0] = k;
    dims[1] = d;
    PyObject *ret = PyArray_SimpleNew(2, dims, typenum);
    float *p_ret = (float *) PyArray_DATA(ret);
    for (int k_i = 0; k_i < k; ++k_i) {
        memcpy(p_ret, cluster_centers[k_i], sizeof(float) * d);
        p_ret += d;
    }

    /* Clean up. */
    Py_DECREF(descr);
    Py_DECREF(X_array);
    free_cluster_centers(k, cluster_centers, d);

    return ret;
}
