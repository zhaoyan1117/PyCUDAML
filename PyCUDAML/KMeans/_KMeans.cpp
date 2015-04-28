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
    int k;
    PyObject *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "iO", &k, &X_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_FLOAT, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (X_array == NULL) {
        Py_XDECREF(X_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(X_array, 0);

    /* Get pointers to the data as C-types. */
    float *X = (float*)PyArray_DATA(X_array);

    /* Call the external C++ function to fit K-Means clustering. */
    float value = kmeans(k, X, N);

    /* Clean up. */
    Py_DECREF(X_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("f", value);
    return ret;
}
