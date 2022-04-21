#include "Python.h"

int Ddb_PyObject_LookupAttrId(PyObject *self, struct _Py_Identifier *pyid, PyObject **pAttrValue){
	*pAttrValue = _PyObject_GetAttrId(self, pyid);
	if (*pAttrValue == NULL) {
		if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
			return -1;
		}
		PyErr_Clear();
	}
	return 0;
}

int Ddb_PyObject_LookupAttr(PyObject *self, PyObject *name, PyObject **pAttrValue){
    *pAttrValue = PyObject_GetAttr(self, name);
    return 0;
}

// void PyErr_Format(PyObject *exception,const char *format,PyObject *name,PyObject *obj){
//         PyErr_Clear();
//         PyErr_Format(exception,
//                      format, name, obj);
// }

int Ddb_PyArg_UnpackStackOrTuple(
    PyObject *const *args,
    Py_ssize_t nargs,
    const char *name,
    Py_ssize_t min,
    Py_ssize_t max,
    PyObject **pmodule_name,
    PyObject **pglobal_name){
    return PyArg_UnpackTuple((PyObject*)args, name,
                        2, 2,
                        pmodule_name, pglobal_name);
}
