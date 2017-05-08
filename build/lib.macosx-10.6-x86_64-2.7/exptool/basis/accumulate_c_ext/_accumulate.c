#include <Python.h>
//#include <numpy/arrayobject.h>
#include "accumulate.h"


/* Docstrings */
static char module_docstring[] =
    "This module provides an interface for calculating chi-squared using C.";
static char r_to_xi_docstring[] = 
	"Transform r to xi";


/* Available functions */
static PyObject *accumulate_r_to_xi(PyObject *self, PyObject *args);


/* Module specification */
static PyMethodDef module_methods[] = {
    {"r_to_xi", accumulate_r_to_xi, METH_VARARGS, r_to_xi_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_accumulate(void)
{
    PyObject *m = Py_InitModule3("_accumulate", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *accumulate_r_to_xi(PyObject *self, PyObject *args)
{
    double r,scale;
    int cmap;
	PyObject * ret;

	if (!PyArg_ParseTuple(args, "did", &r, &cmap, &scale))
	return NULL;

	double result = r_to_xi(r,cmap,scale);

	ret = PyFloat_FromDouble(result);
	return ret;
}
