#include "Python.h"
int Ddb_PyArg_UnpackStackOrTuple(
    PyObject *const *args,
    Py_ssize_t nargs,
    const char *name,
    Py_ssize_t min,
    Py_ssize_t max,
    PyObject **pmodule_name,
    PyObject **pglobal_name){
    return _PyArg_UnpackStack(args, nargs, name,
                        2, 2,
                        pmodule_name, pglobal_name);
}

int Ddb_PyObject_LookupAttrId(PyObject *self, struct _Py_Identifier *pyid, PyObject **pAttrValue){
    return _PyObject_LookupAttrId(self,pyid, pAttrValue);
}

int Ddb_PyObject_LookupAttr(PyObject *self, PyObject *name, PyObject **pAttrValue){
    return _PyObject_LookupAttr(self, name, pAttrValue);
}