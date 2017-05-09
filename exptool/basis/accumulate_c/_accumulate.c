#include <Python.h>
//#include <numpy/arrayobject.h>
#include "accumulate.h"


/* Docstrings */
static char module_docstring[] = "Module to migrate some potential transformations to C";
static char r_to_xi_docstring[] = "Transform r to xi";
static char xi_to_r_docstring[] = "Transform xi to r";
static char d_xi_to_r_docstring[] = "Transform derivative of xi to derivative of r";


/* Available functions */
static PyObject *accumulate_r_to_xi(PyObject *self, PyObject *args);
static PyObject *accumulate_xi_to_r(PyObject *self, PyObject *args);
static PyObject *accumulate_d_xi_to_r(PyObject *self, PyObject *args);


/* Module specification */
static PyMethodDef module_methods[] = {
    {"r_to_xi", accumulate_r_to_xi, METH_VARARGS, r_to_xi_docstring},
    {"xi_to_r", accumulate_xi_to_r, METH_VARARGS, xi_to_r_docstring},
    {"d_xi_to_r", accumulate_d_xi_to_r, METH_VARARGS, d_xi_to_r_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_accumulate_c(void)
{

  (void)Py_InitModule3("_accumulate_c", module_methods, module_docstring);
  //PyObject *m = Py_InitModule3("_accumulate_c", module_methods, module_docstring);
  //if (m == NULL)
  //    return;

    /* Load `numpy` functionality. */
    //import_array();
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

static PyObject *accumulate_xi_to_r(PyObject *self, PyObject *args)
{
    double xi,scale;
    int cmap;
	PyObject * ret;

	if (!PyArg_ParseTuple(args, "did", &xi, &cmap, &scale))
	return NULL;

	double result = xi_to_r(xi,cmap,scale);

	ret = PyFloat_FromDouble(result);
	return ret;
}


static PyObject *accumulate_d_xi_to_r(PyObject *self, PyObject *args)
{
    double xi,scale;
    int cmap;
	PyObject * ret;

	if (!PyArg_ParseTuple(args, "did", &xi, &cmap, &scale))
	return NULL;

	double result = d_xi_to_r(xi,cmap,scale);

	ret = PyFloat_FromDouble(result);
	return ret;
}

