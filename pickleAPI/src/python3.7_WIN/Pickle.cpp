#include "Python.h"
#include "Pickle.h"
#include "structmember.h"
#include "ScalarImp.h"
#include "Util.h"
namespace ddb = dolphindb;
#define PyBytesObject_SIZE (offsetof(PyBytesObject, ob_sval) + 1)
/* Bump this when new opcodes are added to the pickle protocol. */
namespace Pickle{
    enum opcode {
        MARK            = '(',
        STOP            = '.',
        POP             = '0',
        POP_MARK        = '1',
        DUP             = '2',
        FLOAT           = 'F',
        INT             = 'I',
        BININT          = 'J',
        BININT1         = 'K',
        LONG            = 'L',
        BININT2         = 'M',
        NONE            = 'N',
        PERSID          = 'P',
        BINPERSID       = 'Q',
        REDUCE          = 'R',
        STRING          = 'S',
        BINSTRING       = 'T',
        SHORT_BINSTRING = 'U',
        UNICODE         = 'V',
        BINUNICODE      = 'X',
        APPEND          = 'a',
        BUILD           = 'b',
        GLOBAL          = 'c',
        DICT            = 'd',
        EMPTY_DICT      = '}',
        APPENDS         = 'e',
        GET             = 'g',
        BINGET          = 'h',
        INST            = 'i',
        LONG_BINGET     = 'j',
        LIST            = 'l',
        EMPTY_LIST      = ']',
        OBJ             = 'o',
        PUT             = 'p',
        BINPUT          = 'q',
        LONG_BINPUT     = 'r',
        SETITEM         = 's',
        TUPLE           = 't',
        EMPTY_TUPLE     = ')',
        SETITEMS        = 'u',
        BINFLOAT        = 'G',

        /* Protocol 2. */
        PROTO       = '\x80',
        NEWOBJ      = '\x81',
        EXT1        = '\x82',
        EXT2        = '\x83',
        EXT4        = '\x84',
        TUPLE1      = '\x85',
        TUPLE2      = '\x86',
        TUPLE3      = '\x87',
        NEWTRUE     = '\x88',
        NEWFALSE    = '\x89',
        LONG1       = '\x8a',
        LONG4       = '\x8b',

        /* Protocol 3 (Python 3.x) */
        BINBYTES       = 'B',
        SHORT_BINBYTES = 'C',

        /* Protocol 4 */
        SHORT_BINUNICODE = '\x8c',
        BINUNICODE8      = '\x8d',
        BINBYTES8        = '\x8e',
        EMPTY_SET        = '\x8f',
        ADDITEMS         = '\x90',
        FROZENSET        = '\x91',
        NEWOBJ_EX        = '\x92',
        STACK_GLOBAL     = '\x93',
        MEMOIZE          = '\x94',
        FRAME            = '\x95',

        SYMBOL           = '\xf1',
        OBJECTBEGIN      = '\xf2'
    };
}
enum {
    HIGHEST_PROTOCOL = 4,
    DEFAULT_PROTOCOL = 3
};

/* Pickle opcodes. These must be kept updated with pickle.py.
   Extensive docs are in pickletools.py. */


enum {
    /* Keep in synch with pickle.Pickler._BATCHSIZE.  This is how many elements
       batch_list/dict() pumps out before doing APPENDS/SETITEMS.  Nothing will
       break if this gets out of synch with pickle.py, but it's unclear that would
       help anything either. */
    BATCHSIZE = 1000,

    /* Nesting limit until Pickler, when running in "fast mode", starts
       checking for self-referential data-structures. */
    FAST_NESTING_LIMIT = 50,

    /* Initial size of the write buffer of Pickler. */
    WRITE_BUF_SIZE = 4096,

    /* Prefetch size when unpickling (disabled on unpeekable streams) */
    PREFETCH = 8192 * 16,

    FRAME_SIZE_MIN = 4,
    FRAME_SIZE_TARGET = 64 * 1024,
    FRAME_HEADER_SIZE = 9
};

/* State of the pickle module, per PEP 3121. */
typedef struct {
    /* Exception classes for pickle. */
    PyObject *PickleError;
    PyObject *PicklingError;
    PyObject *UnpicklingError;

    /* copyreg.dispatch_table, {type_object: pickling_function} */
    PyObject *dispatch_table;

    /* For the extension opcodes EXT1, EXT2 and EXT4. */

    /* copyreg._extension_registry, {(module_name, function_name): code} */
    PyObject *extension_registry;
    /* copyreg._extension_cache, {code: object} */
    PyObject *extension_cache;
    /* copyreg._inverted_registry, {code: (module_name, function_name)} */
    PyObject *inverted_registry;

    /* Import mappings for compatibility with Python 2.x */

    /* _compat_pickle.NAME_MAPPING,
       {(oldmodule, oldname): (newmodule, newname)} */
    PyObject *name_mapping_2to3;
    /* _compat_pickle.IMPORT_MAPPING, {oldmodule: newmodule} */
    PyObject *import_mapping_2to3;
    /* Same, but with REVERSE_NAME_MAPPING / REVERSE_IMPORT_MAPPING */
    PyObject *name_mapping_3to2;
    PyObject *import_mapping_3to2;

    /* codecs.encode, used for saving bytes in older protocols */
    PyObject *codecs_encode;
    /* builtins.getattr, used for saving nested names with protocol < 4 */
    PyObject *getattr;
    /* functools.partial, used for implementing __newobj_ex__ with protocols
       2 and 3 */
    PyObject *partial;
} PickleState;


/* Given a module object, get its per-module state. */
static PickleState *
_Pickle_GetState(PyObject *module)
{
    return (PickleState *)PyModule_GetState(module);
}

/* Find the module instance imported in the currently running sub-interpreter
   and get its state. */
static PickleState *_Pickle_GetGlobalState(void);

/* Helper for calling a function with a single argument quickly.

   This function steals the reference of the given argument. */
static PyObject *
_Pickle_FastCall(PyObject *func, PyObject *obj)
{
    PyObject *result;

    result = PyObject_CallFunctionObjArgs(func, obj, NULL);
    Py_DECREF(obj);
    return result;
}

/* Retrieve and deconstruct a method for avoiding a reference cycle
   (pickler -> bound method of pickler -> pickler) */
static int
init_method_ref(PyObject *self, _Py_Identifier *name,
                PyObject **method_func, PyObject **method_self)
{
    PyObject *func, *func2;
    int ret;

    /* *method_func and *method_self should be consistent.  All refcount decrements
       should be occurred after setting *method_self and *method_func. */
    ret = _PyObject_LookupAttrId(self, name, &func);
    if (func == NULL) {
        *method_self = NULL;
        Py_CLEAR(*method_func);
        return ret;
    }

    if (PyMethod_Check(func) && PyMethod_GET_SELF(func) == self) {
        /* Deconstruct a bound Python method */
        func2 = PyMethod_GET_FUNCTION(func);
        Py_INCREF(func2);
        *method_self = self; /* borrowed */
        Py_XSETREF(*method_func, func2);
        Py_DECREF(func);
        return 0;
    }
    else {
        *method_self = NULL;
        Py_XSETREF(*method_func, func);
        return 0;
    }
}

/* Bind a method if it was deconstructed */
static PyObject *
reconstruct_method(PyObject *func, PyObject *self)
{
    if (self) {
        return PyMethod_New(func, self);
    }
    else {
        Py_INCREF(func);
        return func;
    }
}

static PyObject *
call_method(PyObject *func, PyObject *self, PyObject *obj)
{
    if (self) {
        return PyObject_CallFunctionObjArgs(func, self, obj, NULL);
    }
    else {
        return PyObject_CallFunctionObjArgs(func, obj, NULL);
    }
}

/* Internal data type used as the unpickling stack. */
typedef struct {
    PyObject_VAR_HEAD
    PyObject **data;
    int mark_set;          /* is MARK set? */
    Py_ssize_t fence;      /* position of top MARK or 0 */
    Py_ssize_t allocated;  /* number of slots in data allocated */
} Pdata;

static void
Pdata_dealloc(Pdata *self)
{
    Py_ssize_t i = Py_SIZE(self);
    while (--i >= 0) {
        Py_DECREF(self->data[i]);
    }
    PyMem_FREE(self->data);
    PyObject_Del(self);
}

static PyTypeObject Pdata_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "_pickle.Pdata",              /*tp_name*/
        sizeof(Pdata),                /*tp_basicsize*/
        sizeof(PyObject *),           /*tp_itemsize*/
        (destructor)Pdata_dealloc,    /*tp_dealloc*/
};

static PyObject *
Pdata_New(void)
{
    Pdata *self;

    if (!(self = PyObject_New(Pdata, &Pdata_Type)))
        return NULL;
    Py_SIZE(self) = 0;
    self->mark_set = 0;
    self->fence = 0;
    self->allocated = 8;
    self->data = (PyObject **)PyMem_MALLOC(self->allocated * sizeof(PyObject *));
    if (self->data)
        return (PyObject *)self;
    Py_DECREF(self);
    return PyErr_NoMemory();
}

/* Retain only the initial clearto items.  If clearto >= the current
 * number of items, this is a (non-erroneous) NOP.
 */
static int
Pdata_clear(Pdata *self, Py_ssize_t clearto)
{
    Py_ssize_t i = Py_SIZE(self);

    assert(clearto >= self->fence);
    if (clearto >= i)
        return 0;

    while (--i >= clearto) {
        Py_CLEAR(self->data[i]);
    }
    Py_SIZE(self) = clearto;
    return 0;
}

static int
Pdata_grow(Pdata *self)
{
    PyObject **data = self->data;
    size_t allocated = (size_t)self->allocated;
    size_t new_allocated;

    new_allocated = (allocated >> 3) + 6;
    /* check for integer overflow */
    if (new_allocated > (size_t)PY_SSIZE_T_MAX - allocated)
        goto nomemory;
    new_allocated += allocated;
    PyMem_RESIZE(data, PyObject *, new_allocated);
    if (data == NULL)
        goto nomemory;

    self->data = data;
    self->allocated = (Py_ssize_t)new_allocated;
    return 0;

    nomemory:
    PyErr_NoMemory();
    return -1;
}

PickleState * _Pickle_GetGlobalState(){
    PyObject * _pic = PyImport_ImportModule("_pickle");
    PickleState *st = _Pickle_GetState(_pic);
    return st;
}

static int
Pdata_stack_underflow(Pdata *self)
{
    PickleState *st = _Pickle_GetGlobalState();
    PyErr_SetString(st->UnpicklingError,
                    self->mark_set ?
                    "unexpected MARK found" :
                    "unpickling stack underflow");
    return -1;
}

/* D is a Pdata*.  Pop the topmost element and store it into V, which
 * must be an lvalue holding PyObject*.  On stack underflow, UnpicklingError
 * is raised and V is set to NULL.
 */
static PyObject *
Pdata_pop(Pdata *self)
{
    if (Py_SIZE(self) <= self->fence) {
        Pdata_stack_underflow(self);
        return NULL;
    }
    return self->data[--Py_SIZE(self)];
}
#define PDATA_POP(D, V) do { (V) = Pdata_pop((D)); } while (0)

static int
Pdata_push(Pdata *self, PyObject *obj)
{
    if (Py_SIZE(self) == self->allocated && Pdata_grow(self) < 0) {
        return -1;
    }
    self->data[Py_SIZE(self)++] = obj;
    return 0;
}

/* Push an object on stack, transferring its ownership to the stack. */
#define PDATA_PUSH(D, O, ER) do {                               \
        if (Pdata_push((D), (O)) < 0) return (ER); } while(0)

/* Push an object on stack, adding a new reference to the object. */
#define PDATA_APPEND(D, O, ER) do {                             \
        Py_INCREF((O));                                         \
        if (Pdata_push((D), (O)) < 0) return (ER); } while(0)

static PyObject *
Pdata_poptuple(Pdata *self, Py_ssize_t start)
{
    PyObject *tuple;
    Py_ssize_t len, i, j;

    if (start < self->fence) {
        Pdata_stack_underflow(self);
        return NULL;
    }
    len = Py_SIZE(self) - start;
    tuple = PyTuple_New(len);
    if (tuple == NULL)
        return NULL;
    for (i = start, j = 0; j < len; i++, j++)
        PyTuple_SET_ITEM(tuple, j, self->data[i]);

    Py_SIZE(self) = start;
    return tuple;
}

static PyObject *
Pdata_poplist(Pdata *self, Py_ssize_t start)
{
    PyObject *list;
    Py_ssize_t len, i, j;

    len = Py_SIZE(self) - start;
    list = PyList_New(len);
    if (list == NULL)
        return NULL;
    for (i = start, j = 0; j < len; i++, j++)
        PyList_SET_ITEM(list, j, self->data[i]);

    Py_SIZE(self) = start;
    return list;
}

typedef struct {
    PyObject *me_key;
    Py_ssize_t me_value;
} PyMemoEntry;

typedef struct {
    size_t mt_mask;
    size_t mt_used;
    size_t mt_allocated;
    PyMemoEntry *mt_table;
} PyMemoTable;


typedef struct UnpicklerObject {
    PyObject_HEAD
    Pdata *stack;               /* Pickle data stack, store unpickled objects. */

    /* The unpickler memo is just an array of PyObject *s. Using a dict
       is unnecessary, since the keys are contiguous ints. */
    PyObject **memo;
    size_t memo_size;       /* Capacity of the memo array */
    size_t memo_len;        /* Number of objects in the memo */

    PyObject *pers_func;        /* persistent_load() method, can be NULL. */
    PyObject *pers_func_self;   /* borrowed reference to self if pers_func
                                   is an unbound method, NULL otherwise */

    Py_buffer buffer;
    char *input_buffer;
    char *input_line;
    Py_ssize_t input_len;
    Py_ssize_t next_read_idx;
    Py_ssize_t prefetched_idx;  /* index of first prefetched byte */

    PyObject *read;             /* read() method of the input stream. */
    PyObject *readline;         /* readline() method of the input stream. */
    PyObject *peek;             /* peek() method of the input stream, or NULL */

    char *encoding;             /* Name of the encoding to be used for
                                   decoding strings pickled using Python
                                   2.x. The default value is "ASCII" */
    char *errors;               /* Name of errors handling scheme to used when
                                   decoding strings. The default value is
                                   "strict". */
    Py_ssize_t *marks;          /* Mark stack, used for unpickling container
                                   objects. */
    Py_ssize_t num_marks;       /* Number of marks in the mark stack. */
    Py_ssize_t marks_size;      /* Current allocated size of the mark stack. */
    int proto;                  /* Protocol of the pickle loaded. */
    int fix_imports;            /* Indicate whether Unpickler should fix
                                   the name of globals pickled by Python 2.x. */
} UnpicklerObject;

typedef struct {
    PyObject_HEAD
    UnpicklerObject *unpickler;
} UnpicklerMemoProxyObject;


PyDoc_STRVAR(_pickle_Unpickler_find_class__doc__,
             "find_class($self, module_name, global_name, /)\n"
             "--\n"
             "\n"
             "Return an object from a specified module.\n"
             "\n"
             "If necessary, the module will be imported. Subclasses may override\n"
             "this method (e.g. to restrict unpickling of arbitrary classes and\n"
             "functions).\n"
             "\n"
             "This method is called whenever a class or a function object is\n"
             "needed.  Both arguments passed are str objects.");

#define _PICKLE_UNPICKLER_FIND_CLASS_METHODDEF    \
    {"find_class", (PyCFunction)_pickle_Unpickler_find_class, METH_FASTCALL, _pickle_Unpickler_find_class__doc__},

static PyObject *
_pickle_Unpickler_find_class_impl(UnpicklerObject *self,
                                  PyObject *module_name,
                                  PyObject *global_name);

static PyObject *
_pickle_Unpickler_find_class(UnpicklerObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    PyObject *return_value = NULL;
    PyObject *module_name;
    PyObject *global_name;

    if (!_PyArg_UnpackStack(args, nargs, "find_class",
                            2, 2,
                            &module_name, &global_name)) {
        goto exit;
    }
    return_value = _pickle_Unpickler_find_class_impl(self, module_name, global_name);

    exit:
    return return_value;
}

PyDoc_STRVAR(_pickle_Unpickler___sizeof____doc__,
             "__sizeof__($self, /)\n"
             "--\n"
             "\n"
             "Returns size in memory, in bytes.");

#define _PICKLE_UNPICKLER___SIZEOF___METHODDEF    \
    {"__sizeof__", (PyCFunction)_pickle_Unpickler___sizeof__, METH_NOARGS, _pickle_Unpickler___sizeof____doc__},

static Py_ssize_t
_pickle_Unpickler___sizeof___impl(UnpicklerObject *self);

static PyObject *
_pickle_Unpickler___sizeof__(UnpicklerObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *return_value = NULL;
    Py_ssize_t _return_value;

    _return_value = _pickle_Unpickler___sizeof___impl(self);
    if ((_return_value == -1) && PyErr_Occurred()) {
        goto exit;
    }
    return_value = PyLong_FromSsize_t(_return_value);

    exit:
    return return_value;
}

PyDoc_STRVAR(_pickle_Unpickler___init____doc__,
             "Unpickler(file, *, fix_imports=True, encoding=\'ASCII\', errors=\'strict\')\n"
             "--\n"
             "\n"
             "This takes a binary file for reading a pickle data stream.\n"
             "\n"
             "The protocol version of the pickle is detected automatically, so no\n"
             "protocol argument is needed.  Bytes past the pickled object\'s\n"
             "representation are ignored.\n"
             "\n"
             "The argument *file* must have two methods, a read() method that takes\n"
             "an integer argument, and a readline() method that requires no\n"
             "arguments.  Both methods should return bytes.  Thus *file* can be a\n"
             "binary file object opened for reading, an io.BytesIO object, or any\n"
             "other custom object that meets this interface.\n"
             "\n"
             "Optional keyword arguments are *fix_imports*, *encoding* and *errors*,\n"
             "which are used to control compatibility support for pickle stream\n"
             "generated by Python 2.  If *fix_imports* is True, pickle will try to\n"
             "map the old Python 2 names to the new names used in Python 3.  The\n"
             "*encoding* and *errors* tell pickle how to decode 8-bit string\n"
             "instances pickled by Python 2; these default to \'ASCII\' and \'strict\',\n"
             "respectively.  The *encoding* can be \'bytes\' to read these 8-bit\n"
             "string instances as bytes objects.");

static int
_pickle_Unpickler___init___impl(UnpicklerObject *self, PyObject *file,
                                int fix_imports, const char *encoding,
                                const char *errors);

static int
_pickle_Unpickler___init__(PyObject *self, PyObject *args, PyObject *kwargs)
{
    int return_value = -1;
    static const char * const _keywords[] = {"file", "fix_imports", "encoding", "errors", NULL};
    static _PyArg_Parser _parser = {"O|$pss:Unpickler", _keywords, 0};
    PyObject *file;
    int fix_imports = 1;
    const char *encoding = "ASCII";
    const char *errors = "strict";

    if (!_PyArg_ParseTupleAndKeywordsFast(args, kwargs, &_parser,
                                          &file, &fix_imports, &encoding, &errors)) {
        goto exit;
    }
    return_value = _pickle_Unpickler___init___impl((UnpicklerObject *)self, file, fix_imports, encoding, errors);

    exit:
    return return_value;
}

PyDoc_STRVAR(_pickle_UnpicklerMemoProxy_clear__doc__,
             "clear($self, /)\n"
             "--\n"
             "\n"
             "Remove all items from memo.");

#define _PICKLE_UNPICKLERMEMOPROXY_CLEAR_METHODDEF    \
    {"clear", (PyCFunction)_pickle_UnpicklerMemoProxy_clear, METH_NOARGS, _pickle_UnpicklerMemoProxy_clear__doc__},

static PyObject *
_pickle_UnpicklerMemoProxy_clear_impl(UnpicklerMemoProxyObject *self);

static PyObject *
_pickle_UnpicklerMemoProxy_clear(UnpicklerMemoProxyObject *self, PyObject *Py_UNUSED(ignored))
{
    return _pickle_UnpicklerMemoProxy_clear_impl(self);
}

PyDoc_STRVAR(_pickle_UnpicklerMemoProxy_copy__doc__,
             "copy($self, /)\n"
             "--\n"
             "\n"
             "Copy the memo to a new object.");

#define _PICKLE_UNPICKLERMEMOPROXY_COPY_METHODDEF    \
    {"copy", (PyCFunction)_pickle_UnpicklerMemoProxy_copy, METH_NOARGS, _pickle_UnpicklerMemoProxy_copy__doc__},

static PyObject *
_pickle_UnpicklerMemoProxy_copy_impl(UnpicklerMemoProxyObject *self);

static PyObject *
_pickle_UnpicklerMemoProxy_copy(UnpicklerMemoProxyObject *self, PyObject *Py_UNUSED(ignored))
{
    return _pickle_UnpicklerMemoProxy_copy_impl(self);
}

PyDoc_STRVAR(_pickle_UnpicklerMemoProxy___reduce____doc__,
             "__reduce__($self, /)\n"
             "--\n"
             "\n"
             "Implement pickling support.");

#define _PICKLE_UNPICKLERMEMOPROXY___REDUCE___METHODDEF    \
    {"__reduce__", (PyCFunction)_pickle_UnpicklerMemoProxy___reduce__, METH_NOARGS, _pickle_UnpicklerMemoProxy___reduce____doc__},

static PyObject *
_pickle_UnpicklerMemoProxy___reduce___impl(UnpicklerMemoProxyObject *self);

static PyObject *
_pickle_UnpicklerMemoProxy___reduce__(UnpicklerMemoProxyObject *self, PyObject *Py_UNUSED(ignored))
{
    return _pickle_UnpicklerMemoProxy___reduce___impl(self);
}

static int
bad_readline(void)
{
    PickleState *st = _Pickle_GetGlobalState();
    PyErr_SetString(st->UnpicklingError, "pickle data was truncated");
    return -1;
}

static int
_Unpickler_SkipConsumed(UnpicklerObject *self)
{
    Py_ssize_t consumed;
    PyObject *r;

    consumed = self->next_read_idx - self->prefetched_idx;
    if (consumed <= 0)
        return 0;

    assert(self->peek);  /* otherwise we did something wrong */
    /* This makes a useless copy... */
    r = PyObject_CallFunction(self->read, "n", consumed);
    if (r == NULL)
        return -1;
    Py_DECREF(r);

    self->prefetched_idx = self->next_read_idx;
    return 0;
}

static const Py_ssize_t READ_WHOLE_LINE = -1;

/* Returns -1 (with an exception set) on failure, 0 on success. The memo array
   will be modified in place. */
static int
_Unpickler_ResizeMemoList(UnpicklerObject *self, size_t new_size)
{
    size_t i;

    assert(new_size > self->memo_size);

    PyObject **memo_new = self->memo;
    PyMem_RESIZE(memo_new, PyObject *, new_size);
    if (memo_new == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    self->memo = memo_new;
    for (i = self->memo_size; i < new_size; i++)
        self->memo[i] = NULL;
    self->memo_size = new_size;
    return 0;
}

/* Returns NULL if idx is out of bounds. */
static PyObject *
_Unpickler_MemoGet(UnpicklerObject *self, size_t idx)
{
    if (idx >= self->memo_size)
        return NULL;

    return self->memo[idx];
}

/* Returns -1 (with an exception set) on failure, 0 on success.
   This takes its own reference to `value`. */
static int
_Unpickler_MemoPut(UnpicklerObject *self, size_t idx, PyObject *value)
{
    PyObject *old_item;

    if (idx >= self->memo_size) {
        if (_Unpickler_ResizeMemoList(self, idx * 2) < 0)
            return -1;
        assert(idx < self->memo_size);
    }
    Py_INCREF(value);
    old_item = self->memo[idx];
    self->memo[idx] = value;
    if (old_item != NULL) {
        Py_DECREF(old_item);
    }
    else {
        self->memo_len++;
    }
    return 0;
}

static PyObject **
_Unpickler_NewMemo(Py_ssize_t new_size)
{
    PyObject **memo = PyMem_NEW(PyObject *, new_size);
    if (memo == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    memset(memo, 0, new_size * sizeof(PyObject *));
    return memo;
}

/* Free the unpickler's memo, taking care to decref any items left in it. */
static void
_Unpickler_MemoCleanup(UnpicklerObject *self)
{
    Py_ssize_t i;
    PyObject **memo = self->memo;

    if (self->memo == NULL)
        return;
    self->memo = NULL;
    i = self->memo_size;
    while (--i >= 0) {
        Py_XDECREF(memo[i]);
    }
    PyMem_FREE(memo);
}

/* Returns -1 (with an exception set) on failure, 0 on success. This may
   be called once on a freshly created Pickler. */
static int
_Unpickler_SetInputStream(UnpicklerObject *self, PyObject *file)
{
    _Py_IDENTIFIER(peek);
    _Py_IDENTIFIER(read);
    _Py_IDENTIFIER(readline);

    if (_PyObject_LookupAttrId(file, &PyId_peek, &self->peek) < 0) {
        return -1;
    }
    (void)_PyObject_LookupAttrId(file, &PyId_read, &self->read);
    (void)_PyObject_LookupAttrId(file, &PyId_readline, &self->readline);
    if (self->readline == NULL || self->read == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "file must have 'read' and 'readline' attributes");
        }
        Py_CLEAR(self->read);
        Py_CLEAR(self->readline);
        Py_CLEAR(self->peek);
        return -1;
    }
    return 0;
}

/* Returns -1 (with an exception set) on failure, 0 on success. This may
   be called once on a freshly created Pickler. */
static int
_Unpickler_SetInputEncoding(UnpicklerObject *self,
                            const char *encoding,
                            const char *errors)
{
    if (encoding == NULL)
        encoding = "ASCII";
    if (errors == NULL)
        errors = "strict";
    self->encoding = _PyMem_Strdup(encoding);
    self->errors = _PyMem_Strdup(errors);
    if (self->encoding == NULL || self->errors == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}


static PyObject *
get_dotted_path(PyObject *obj, PyObject *name)
{
    _Py_static_string(PyId_dot, ".");
    PyObject *dotted_path;
    Py_ssize_t i, n;

    dotted_path = PyUnicode_Split(name, _PyUnicode_FromId(&PyId_dot), -1);
    if (dotted_path == NULL)
        return NULL;
    n = PyList_GET_SIZE(dotted_path);
    assert(n >= 1);
    for (i = 0; i < n; i++) {
        PyObject *subpath = PyList_GET_ITEM(dotted_path, i);
        if (_PyUnicode_EqualToASCIIString(subpath, "<locals>")) {
            if (obj == NULL)
                PyErr_Format(PyExc_AttributeError,
                             "Can't pickle local object %R", name);
            else
                PyErr_Format(PyExc_AttributeError,
                             "Can't pickle local attribute %R on %R", name, obj);
            Py_DECREF(dotted_path);
            return NULL;
        }
    }
    return dotted_path;
}

static PyObject *
get_deep_attribute(PyObject *obj, PyObject *names, PyObject **pparent)
{
    Py_ssize_t i, n;
    PyObject *parent = NULL;

    assert(PyList_CheckExact(names));
    Py_INCREF(obj);
    n = PyList_GET_SIZE(names);
    for (i = 0; i < n; i++) {
        PyObject *name = PyList_GET_ITEM(names, i);
        Py_XDECREF(parent);
        parent = obj;
        (void)_PyObject_LookupAttr(parent, name, &obj);
        if (obj == NULL) {
            Py_DECREF(parent);
            return NULL;
        }
    }
    if (pparent != NULL)
        *pparent = parent;
    else
        Py_XDECREF(parent);
    return obj;
}

static PyObject *
getattribute(PyObject *obj, PyObject *name, int allow_qualname)
{
    PyObject *dotted_path, *attr;

    if (allow_qualname) {
        dotted_path = get_dotted_path(obj, name);
        if (dotted_path == NULL)
            return NULL;
        attr = get_deep_attribute(obj, dotted_path, NULL);
        Py_DECREF(dotted_path);
    }
    else {
        (void)_PyObject_LookupAttr(obj, name, &attr);
    }
    if (attr == NULL && !PyErr_Occurred()) {
        PyErr_Format(PyExc_AttributeError,
                     "Can't get attribute %R on %R", name, obj);
    }
    return attr;
}

static PyObject *
find_class(UnpicklerObject *self, PyObject *module_name, PyObject *global_name)
{
    //_Py_IDENTIFIER(find_class);

    //return _PyObject_CallMethodIdObjArgs((PyObject *)self, &PyId_find_class,
    //                                     module_name, global_name, NULL);
    return _pickle_Unpickler_find_class_impl(self, module_name, global_name);
}

static Py_ssize_t
marker(UnpicklerObject *self)
{
    Py_ssize_t mark;

    if (self->num_marks < 1) {
        PickleState *st = _Pickle_GetGlobalState();
        PyErr_SetString(st->UnpicklingError, "could not find MARK");
        return -1;
    }

    mark = self->marks[--self->num_marks];
    self->stack->mark_set = self->num_marks != 0;
    self->stack->fence = self->num_marks ?
                         self->marks[self->num_marks - 1] : 0;
    return mark;
}
/* The name of find_class() is misleading. In newer pickle protocols, this
   function is used for loading any global (i.e., functions), not just
   classes. The name is kept only for backward compatibility. */

/*[clinic input]

_pickle.Unpickler.find_class

  module_name: object
  global_name: object
  /

Return an object from a specified module.

If necessary, the module will be imported. Subclasses may override
this method (e.g. to restrict unpickling of arbitrary classes and
functions).

This method is called whenever a class or a function object is
needed.  Both arguments passed are str objects.
[clinic start generated code]*/

static PyObject *
_pickle_Unpickler_find_class_impl(UnpicklerObject *self,
                                  PyObject *module_name,
                                  PyObject *global_name)
/*[clinic end generated code: output=becc08d7f9ed41e3 input=e2e6a865de093ef4]*/
{
    PyObject *global;
    PyObject *module;

    /* Try to map the old names used in Python 2.x to the new ones used in
       Python 3.x.  We do this only with old pickle protocols and when the
       user has not disabled the feature. */
    if (self->proto < 3 && self->fix_imports) {
        PyObject *key;
        PyObject *item;
        PickleState *st = _Pickle_GetGlobalState();

        /* Check if the global (i.e., a function or a class) was renamed
           or moved to another module. */
        key = PyTuple_Pack(2, module_name, global_name);
        if (key == NULL)
            return NULL;
        item = PyDict_GetItemWithError(st->name_mapping_2to3, key);
        Py_DECREF(key);
        if (item) {
            if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2) {
                PyErr_Format(PyExc_RuntimeError,
                             "_compat_pickle.NAME_MAPPING values should be "
                             "2-tuples, not %.200s", Py_TYPE(item)->tp_name);
                return NULL;
            }
            module_name = PyTuple_GET_ITEM(item, 0);
            global_name = PyTuple_GET_ITEM(item, 1);
            if (!PyUnicode_Check(module_name) ||
                !PyUnicode_Check(global_name)) {
                PyErr_Format(PyExc_RuntimeError,
                             "_compat_pickle.NAME_MAPPING values should be "
                             "pairs of str, not (%.200s, %.200s)",
                             Py_TYPE(module_name)->tp_name,
                             Py_TYPE(global_name)->tp_name);
                return NULL;
            }
        }
        else if (PyErr_Occurred()) {
            return NULL;
        }
        else {
            /* Check if the module was renamed. */
            item = PyDict_GetItemWithError(st->import_mapping_2to3, module_name);
            if (item) {
                if (!PyUnicode_Check(item)) {
                    PyErr_Format(PyExc_RuntimeError,
                                 "_compat_pickle.IMPORT_MAPPING values should be "
                                 "strings, not %.200s", Py_TYPE(item)->tp_name);
                    return NULL;
                }
                module_name = item;
            }
            else if (PyErr_Occurred()) {
                return NULL;
            }
        }
    }

    /*
     * we don't use PyImport_GetModule here, because it can return partially-
     * initialised modules, which then cause the getattribute to fail.
     */
    module = PyImport_Import(module_name);
    if (module == NULL) {
        return NULL;
    }
    global = getattribute(module, global_name, self->proto >= 4);
    Py_DECREF(module);
    return global;
}

/*[clinic input]

_pickle.Unpickler.__sizeof__ -> Py_ssize_t

Returns size in memory, in bytes.
[clinic start generated code]*/

static Py_ssize_t
_pickle_Unpickler___sizeof___impl(UnpicklerObject *self)
/*[clinic end generated code: output=119d9d03ad4c7651 input=13333471fdeedf5e]*/
{
    Py_ssize_t res;

    res = _PyObject_SIZE(Py_TYPE(self));
    if (self->memo != NULL)
        res += self->memo_size * sizeof(PyObject *);
    if (self->marks != NULL)
        res += self->marks_size * sizeof(Py_ssize_t);
    if (self->input_line != NULL)
        res += strlen(self->input_line) + 1;
    if (self->encoding != NULL)
        res += strlen(self->encoding) + 1;
    if (self->errors != NULL)
        res += strlen(self->errors) + 1;
    return res;
}

static struct PyMethodDef Unpickler_methods[] = {
        //_PICKLE_UNPICKLER_LOAD_METHODDEF
        _PICKLE_UNPICKLER_FIND_CLASS_METHODDEF
        _PICKLE_UNPICKLER___SIZEOF___METHODDEF
        {NULL, NULL}                /* sentinel */
};

static void
Unpickler_dealloc(UnpicklerObject *self)
{
    PyObject_GC_UnTrack((PyObject *)self);
    Py_XDECREF(self->readline);
    Py_XDECREF(self->read);
    Py_XDECREF(self->peek);
    Py_XDECREF(self->stack);
    Py_XDECREF(self->pers_func);
    if (self->buffer.buf != NULL) {
        PyBuffer_Release(&self->buffer);
        self->buffer.buf = NULL;
    }

    _Unpickler_MemoCleanup(self);
    PyMem_Free(self->marks);
    PyMem_Free(self->input_line);
    PyMem_Free(self->encoding);
    PyMem_Free(self->errors);

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
Unpickler_traverse(UnpicklerObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->readline);
    Py_VISIT(self->read);
    Py_VISIT(self->peek);
    Py_VISIT(self->stack);
    Py_VISIT(self->pers_func);
    return 0;
}

static int
Unpickler_clear(UnpicklerObject *self)
{
    Py_CLEAR(self->readline);
    Py_CLEAR(self->read);
    Py_CLEAR(self->peek);
    Py_CLEAR(self->stack);
    Py_CLEAR(self->pers_func);
    if (self->buffer.buf != NULL) {
        PyBuffer_Release(&self->buffer);
        self->buffer.buf = NULL;
    }

    _Unpickler_MemoCleanup(self);
    PyMem_Free(self->marks);
    self->marks = NULL;
    PyMem_Free(self->input_line);
    self->input_line = NULL;
    PyMem_Free(self->encoding);
    self->encoding = NULL;
    PyMem_Free(self->errors);
    self->errors = NULL;

    return 0;
}

/*[clinic input]

_pickle.Unpickler.__init__

  file: object
  *
  fix_imports: bool = True
  encoding: str = 'ASCII'
  errors: str = 'strict'

This takes a binary file for reading a pickle data stream.

The protocol version of the pickle is detected automatically, so no
protocol argument is needed.  Bytes past the pickled object's
representation are ignored.

The argument *file* must have two methods, a read() method that takes
an integer argument, and a readline() method that requires no
arguments.  Both methods should return bytes.  Thus *file* can be a
binary file object opened for reading, an io.BytesIO object, or any
other custom object that meets this interface.

Optional keyword arguments are *fix_imports*, *encoding* and *errors*,
which are used to control compatibility support for pickle stream
generated by Python 2.  If *fix_imports* is True, pickle will try to
map the old Python 2 names to the new names used in Python 3.  The
*encoding* and *errors* tell pickle how to decode 8-bit string
instances pickled by Python 2; these default to 'ASCII' and 'strict',
respectively.  The *encoding* can be 'bytes' to read these 8-bit
string instances as bytes objects.
[clinic start generated code]*/

static int
_pickle_Unpickler___init___impl(UnpicklerObject *self, PyObject *file,
                                int fix_imports, const char *encoding,
                                const char *errors)
/*[clinic end generated code: output=e2c8ce748edc57b0 input=f9b7da04f5f4f335]*/
{
    _Py_IDENTIFIER(persistent_load);

    /* In case of multiple __init__() calls, clear previous content. */
    if (self->read != NULL)
        (void)Unpickler_clear(self);

    if (_Unpickler_SetInputStream(self, file) < 0)
        return -1;

    if (_Unpickler_SetInputEncoding(self, encoding, errors) < 0)
        return -1;

    self->fix_imports = fix_imports;

    if (init_method_ref((PyObject *)self, &PyId_persistent_load,
                        &self->pers_func, &self->pers_func_self) < 0)
    {
        return -1;
    }

    self->stack = (Pdata *)Pdata_New();
    if (self->stack == NULL)
        return -1;

    self->memo_size = 32;
    self->memo = _Unpickler_NewMemo(self->memo_size);
    if (self->memo == NULL)
        return -1;

    self->proto = 0;

    return 0;
}


/* Define a proxy object for the Unpickler's internal memo object. This is to
 * avoid breaking code like:
 *  unpickler.memo.clear()
 * and
 *  unpickler.memo = saved_memo
 * Is this a good idea? Not really, but we don't want to break code that uses
 * it. Note that we don't implement the entire mapping API here. This is
 * intentional, as these should be treated as black-box implementation details.
 *
 * We do, however, have to implement pickling/unpickling support because of
 * real-world code like cvs2svn.
 */

/*[clinic input]
_pickle.UnpicklerMemoProxy.clear

Remove all items from memo.
[clinic start generated code]*/

static PyObject *
_pickle_UnpicklerMemoProxy_clear_impl(UnpicklerMemoProxyObject *self)
/*[clinic end generated code: output=d20cd43f4ba1fb1f input=b1df7c52e7afd9bd]*/
{
    _Unpickler_MemoCleanup(self->unpickler);
    self->unpickler->memo = _Unpickler_NewMemo(self->unpickler->memo_size);
    if (self->unpickler->memo == NULL)
        return NULL;
    Py_RETURN_NONE;
}

/*[clinic input]
_pickle.UnpicklerMemoProxy.copy

Copy the memo to a new object.
[clinic start generated code]*/

static PyObject *
_pickle_UnpicklerMemoProxy_copy_impl(UnpicklerMemoProxyObject *self)
/*[clinic end generated code: output=e12af7e9bc1e4c77 input=97769247ce032c1d]*/
{
    size_t i;
    PyObject *new_memo = PyDict_New();
    if (new_memo == NULL)
        return NULL;

    for (i = 0; i < self->unpickler->memo_size; i++) {
        int status;
        PyObject *key, *value;

        value = self->unpickler->memo[i];
        if (value == NULL)
            continue;

        key = PyLong_FromSsize_t(i);
        if (key == NULL)
            goto error;
        status = PyDict_SetItem(new_memo, key, value);
        Py_DECREF(key);
        if (status < 0)
            goto error;
    }
    return new_memo;

    error:
    Py_DECREF(new_memo);
    return NULL;
}

/*[clinic input]
_pickle.UnpicklerMemoProxy.__reduce__

Implement pickling support.
[clinic start generated code]*/

static PyObject *
_pickle_UnpicklerMemoProxy___reduce___impl(UnpicklerMemoProxyObject *self)
/*[clinic end generated code: output=6da34ac048d94cca input=6920862413407199]*/
{
    PyObject *reduce_value;
    PyObject *constructor_args;
    PyObject *contents = _pickle_UnpicklerMemoProxy_copy_impl(self);
    if (contents == NULL)
        return NULL;

    reduce_value = PyTuple_New(2);
    if (reduce_value == NULL) {
        Py_DECREF(contents);
        return NULL;
    }
    constructor_args = PyTuple_New(1);
    if (constructor_args == NULL) {
        Py_DECREF(contents);
        Py_DECREF(reduce_value);
        return NULL;
    }
    PyTuple_SET_ITEM(constructor_args, 0, contents);
    Py_INCREF((PyObject *)&PyDict_Type);
    PyTuple_SET_ITEM(reduce_value, 0, (PyObject *)&PyDict_Type);
    PyTuple_SET_ITEM(reduce_value, 1, constructor_args);
    return reduce_value;
}

static PyMethodDef unpicklerproxy_methods[] = {
        _PICKLE_UNPICKLERMEMOPROXY_CLEAR_METHODDEF
        _PICKLE_UNPICKLERMEMOPROXY_COPY_METHODDEF
        _PICKLE_UNPICKLERMEMOPROXY___REDUCE___METHODDEF
        {NULL, NULL}    /* sentinel */
};

static void
UnpicklerMemoProxy_dealloc(UnpicklerMemoProxyObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->unpickler);
    PyObject_GC_Del((PyObject *)self);
}

static int
UnpicklerMemoProxy_traverse(UnpicklerMemoProxyObject *self,
                            visitproc visit, void *arg)
{
    Py_VISIT(self->unpickler);
    return 0;
}

static int
UnpicklerMemoProxy_clear(UnpicklerMemoProxyObject *self)
{
    Py_CLEAR(self->unpickler);
    return 0;
}

static PyTypeObject UnpicklerMemoProxyType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "_pickle.UnpicklerMemoProxy",               /*tp_name*/
        sizeof(UnpicklerMemoProxyObject),           /*tp_basicsize*/
        0,
        (destructor)UnpicklerMemoProxy_dealloc,     /* tp_dealloc */
        0,                                          /* tp_print */
        0,                                          /* tp_getattr */
        0,                                          /* tp_setattr */
        0,                                          /* tp_compare */
        0,                                          /* tp_repr */
        0,                                          /* tp_as_number */
        0,                                          /* tp_as_sequence */
        0,                                          /* tp_as_mapping */
        PyObject_HashNotImplemented,                /* tp_hash */
        0,                                          /* tp_call */
        0,                                          /* tp_str */
        PyObject_GenericGetAttr,                    /* tp_getattro */
        PyObject_GenericSetAttr,                    /* tp_setattro */
        0,                                          /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
        0,                                          /* tp_doc */
        (traverseproc)UnpicklerMemoProxy_traverse,  /* tp_traverse */
        (inquiry)UnpicklerMemoProxy_clear,          /* tp_clear */
        0,                                          /* tp_richcompare */
        0,                                          /* tp_weaklistoffset */
        0,                                          /* tp_iter */
        0,                                          /* tp_iternext */
        unpicklerproxy_methods,                     /* tp_methods */
};

static PyObject *
UnpicklerMemoProxy_New(UnpicklerObject *unpickler)
{
    UnpicklerMemoProxyObject *self;

    self = PyObject_GC_New(UnpicklerMemoProxyObject,
                           &UnpicklerMemoProxyType);
    if (self == NULL)
        return NULL;
    Py_INCREF(unpickler);
    self->unpickler = unpickler;
    PyObject_GC_Track(self);
    return (PyObject *)self;
}

/*****************************************************************************/


static PyObject *
Unpickler_get_memo(UnpicklerObject *self, void *Py_UNUSED(ignored))
{
    return UnpicklerMemoProxy_New(self);
}

static int
Unpickler_set_memo(UnpicklerObject *self, PyObject *obj, void *Py_UNUSED(ignored))
{
    PyObject **new_memo;
    size_t new_memo_size = 0;

    if (obj == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "attribute deletion is not supported");
        return -1;
    }

    if (Py_TYPE(obj) == &UnpicklerMemoProxyType) {
        UnpicklerObject *unpickler =
                ((UnpicklerMemoProxyObject *)obj)->unpickler;

        new_memo_size = unpickler->memo_size;
        new_memo = _Unpickler_NewMemo(new_memo_size);
        if (new_memo == NULL)
            return -1;

        for (size_t i = 0; i < new_memo_size; i++) {
            Py_XINCREF(unpickler->memo[i]);
            new_memo[i] = unpickler->memo[i];
        }
    }
    else if (PyDict_Check(obj)) {
        Py_ssize_t i = 0;
        PyObject *key, *value;

        new_memo_size = PyDict_GET_SIZE(obj);
        new_memo = _Unpickler_NewMemo(new_memo_size);
        if (new_memo == NULL)
            return -1;

        while (PyDict_Next(obj, &i, &key, &value)) {
            Py_ssize_t idx;
            if (!PyLong_Check(key)) {
                PyErr_SetString(PyExc_TypeError,
                                "memo key must be integers");
                goto error;
            }
            idx = PyLong_AsSsize_t(key);
            if (idx == -1 && PyErr_Occurred())
                goto error;
            if (idx < 0) {
                PyErr_SetString(PyExc_ValueError,
                                "memo key must be positive integers.");
                goto error;
            }
            if (_Unpickler_MemoPut(self, idx, value) < 0)
                goto error;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "'memo' attribute must be an UnpicklerMemoProxy object "
                     "or dict, not %.200s", Py_TYPE(obj)->tp_name);
        return -1;
    }

    _Unpickler_MemoCleanup(self);
    self->memo_size = new_memo_size;
    self->memo = new_memo;

    return 0;

    error:
    if (new_memo_size) {
        for (size_t i = new_memo_size - 1; i != SIZE_MAX; i--) {
            Py_XDECREF(new_memo[i]);
        }
        PyMem_FREE(new_memo);
    }
    return -1;
}

static PyObject *
Unpickler_get_persload(UnpicklerObject *self, void *Py_UNUSED(ignored))
{
    if (self->pers_func == NULL) {
        PyErr_SetString(PyExc_AttributeError, "persistent_load");
        return NULL;
    }
    return reconstruct_method(self->pers_func, self->pers_func_self);
}

static int
Unpickler_set_persload(UnpicklerObject *self, PyObject *value, void *Py_UNUSED(ignored))
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "attribute deletion is not supported");
        return -1;
    }
    if (!PyCallable_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "persistent_load must be a callable taking "
                        "one argument");
        return -1;
    }

    self->pers_func_self = NULL;
    Py_INCREF(value);
    Py_XSETREF(self->pers_func, value);

    return 0;
}

static PyGetSetDef Unpickler_getsets[] = {
        {"memo", (getter)Unpickler_get_memo, (setter)Unpickler_set_memo},
        {"persistent_load", (getter)Unpickler_get_persload,
                                             (setter)Unpickler_set_persload},
        {NULL}
};

static PyTypeObject Unpickler_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "_pickle.Unpickler",                /*tp_name*/
        sizeof(UnpicklerObject),            /*tp_basicsize*/
        0,                                  /*tp_itemsize*/
        (destructor)Unpickler_dealloc,      /*tp_dealloc*/
        0,                                  /*tp_print*/
        0,                                  /*tp_getattr*/
        0,                                  /*tp_setattr*/
        0,                                  /*tp_reserved*/
        0,                                  /*tp_repr*/
        0,                                  /*tp_as_number*/
        0,                                  /*tp_as_sequence*/
        0,                                  /*tp_as_mapping*/
        0,                                  /*tp_hash*/
        0,                                  /*tp_call*/
        0,                                  /*tp_str*/
        0,                                  /*tp_getattro*/
        0,                                  /*tp_setattro*/
        0,                                  /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
        _pickle_Unpickler___init____doc__,  /*tp_doc*/
        (traverseproc)Unpickler_traverse,   /*tp_traverse*/
        (inquiry)Unpickler_clear,           /*tp_clear*/
        0,                                  /*tp_richcompare*/
        0,                                  /*tp_weaklistoffset*/
        0,                                  /*tp_iter*/
        0,                                  /*tp_iternext*/
        Unpickler_methods,                  /*tp_methods*/
        0,                                  /*tp_members*/
        Unpickler_getsets,                  /*tp_getset*/
        0,                                  /*tp_base*/
        0,                                  /*tp_dict*/
        0,                                  /*tp_descr_get*/
        0,                                  /*tp_descr_set*/
        0,                                  /*tp_dictoffset*/
        _pickle_Unpickler___init__,         /*tp_init*/
        PyType_GenericAlloc,                /*tp_alloc*/
        PyType_GenericNew,                  /*tp_new*/
        PyObject_GC_Del,                    /*tp_free*/
        0,                                  /*tp_is_gc*/
};

static UnpicklerObject *
_Unpickler_New(void)
{
    UnpicklerObject *self;

    self = PyObject_GC_New(UnpicklerObject, &Unpickler_Type);
    if (self == NULL)
        return NULL;

    self->pers_func = NULL;
    self->input_buffer = NULL;
    self->input_line = NULL;
    self->input_len = 0;
    self->next_read_idx = 0;
    self->prefetched_idx = 0;
    self->read = NULL;
    self->readline = NULL;
    self->peek = NULL;
    self->encoding = NULL;
    self->errors = NULL;
    self->marks = NULL;
    self->num_marks = 0;
    self->marks_size = 0;
    self->proto = 0;
    self->fix_imports = 0;
    memset(&self->buffer, 0, sizeof(Py_buffer));
    self->memo_size = 32;
    self->memo_len = 0;
    self->memo = _Unpickler_NewMemo(self->memo_size);
    self->stack = (Pdata *)Pdata_New();

    if (self->memo == NULL || self->stack == NULL) {
        Py_DECREF(self);
        return NULL;
    }

    PyObject_GC_Track(self);
    return self;
}

static Py_ssize_t
calc_binsize(char *bytes, int nbytes)
{
    unsigned char *s = (unsigned char *)bytes;
    int i;
    size_t x = 0;

    if (nbytes > (int)sizeof(size_t)) {
        /* Check for integer overflow.  BINBYTES8 and BINUNICODE8 opcodes
         * have 64-bit size that can't be represented on 32-bit platform.
         */
        for (i = (int)sizeof(size_t); i < nbytes; i++) {
            if (s[i])
                return -1;
        }
        nbytes = (int)sizeof(size_t);
    }
    for (i = 0; i < nbytes; i++) {
        x |= (size_t) s[i] << (8 * i);
    }

    if (x > PY_SSIZE_T_MAX)
        return -1;
    else
        return (Py_ssize_t) x;
}

static long
calc_binint(char *bytes, int nbytes)
{
    unsigned char *s = (unsigned char *)bytes;
    Py_ssize_t i;
    long x = 0;

    for (i = 0; i < nbytes; i++) {
        x |= (long)s[i] << (8 * i);
    }

    /* Unlike BININT1 and BININT2, BININT (more accurately BININT4)
     * is signed, so on a box with longs bigger than 4 bytes we need
     * to extend a BININT's sign bit to the full width.
     */
    if (SIZEOF_LONG > 4 && nbytes == 4) {
        x |= -(x & (1L << 31));
    }

    return x;
}

namespace dolphindb{
    PickleUnmarshall::PickleUnmarshall(const DataInputStreamSP &in) : obj_(NULL), in_(in), frame_(nullptr), frameIdx_(0), frameLen_(0){
        unpickler_ = _Unpickler_New();
        if (unpickler_ == NULL)
            throw RuntimeException("Unpickler initialize error!");
        if (_Unpickler_SetInputEncoding(unpickler_, "ASCII", "strict") < 0)
            throw RuntimeException("Unpickler SetInputEncoding error!");
        unpickler_->fix_imports = 1;
    }

    int PickleUnmarshall::load_none() {
        PDATA_APPEND(unpickler_->stack, Py_None, -1);
        return 0;
    }

    int PickleUnmarshall::load_int(IO_ERR& ret) {
        PyObject *value;
        char *endptr;
        Py_ssize_t len;
        long x;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();

        errno = 0;
        /* XXX: Should the base argument of strtol() be explicitly set to 10?
           XXX(avassalotti): Should this uses PyOS_strtol()? */
        x = strtol(tem.c_str(), &endptr, 0);

        if (errno || (*endptr != '\n' && *endptr != '\0')) {
            /* Hm, maybe we've got something long.  Let's try reading
             * it as a Python int object. */
            errno = 0;
            /* XXX: Same thing about the base here. */
            value = PyLong_FromString(tem.c_str(), NULL, 0);
            if (value == NULL) {
                PyErr_SetString(PyExc_ValueError,
                                "could not convert string to int");
                return -1;
            }
        }
        else {
            if (len == 3 && (x == 0 || x == 1)) {
                if ((value = PyBool_FromLong(x)) == NULL)
                    return -1;
            }
            else {
                if ((value = PyLong_FromLong(x)) == NULL)
                    return -1;
            }
        }

        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_bool(PyObject *boolean) {
        assert(boolean == Py_True || boolean == Py_False);
        PDATA_APPEND(unpickler_->stack, boolean, -1);
        return 0;
    }

    int PickleUnmarshall::load_binintx(char *s, size_t size) {
        PyObject *value;
        long x = calc_binint(s, size);
        if ((value = PyLong_FromLong(x)) == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_binint(IO_ERR& ret) {
        char *s;
        if(frameLen_ - frameIdx_ < 4) {
            if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                return -1;
            return load_binintx(shortBuf_, 4);
        }
        *(&s) = frame_ + frameIdx_;
        frameIdx_ += 4;
        return load_binintx(s, 4);
    }

    int PickleUnmarshall::load_binint1(IO_ERR& ret) {
        char *s;
        if(frameLen_ - frameIdx_ < 1) {
            if((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
                return -1;
            return load_binintx(shortBuf_, 1);
        }
        *(&s) = frame_ + frameIdx_;
        frameIdx_ += 1;
        return load_binintx(s, 1);
    }

    int PickleUnmarshall::load_binint2(IO_ERR& ret) {
        char *s;
        if(frameLen_ - frameIdx_ < 2) {
            if((ret = in_->readBytes(shortBuf_, 2, false)) != OK)
                return -1;
            return load_binintx(shortBuf_, 2);
        }
        *(&s) = frame_ + frameIdx_;
        frameIdx_ += 2;
        return load_binintx(s, 2);
    }

    int PickleUnmarshall::load_long(IO_ERR& ret) {
        PyObject *value;
        Py_ssize_t len;

        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();

        /* s[len-2] will usually be 'L' (and s[len-1] is '\n'); we need to remove
           the 'L' before calling PyLong_FromString.  In order to maintain
           compatibility with Python 3.0.0, we don't actually *require*
           the 'L' to be present. */
        if (tem[len-2] == 'L')
            tem[len-2] = '\0';
        /* XXX: Should the base argument explicitly set to 10? */
        value = PyLong_FromString(tem.c_str(), NULL, 0);
        if (value == NULL)
            return -1;

        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_counted_long(size_t size, IO_ERR& ret) {
        PyObject *value;
        char *nbytes;
        char *pdata;

        assert(size == 1 || size == 4);
        if(frameLen_ - frameIdx_ < size){
            if((ret = in_->readBytes(shortBuf_, size, false)) != OK)
                return -1;
            *(&nbytes) = shortBuf_;
        } else {
            *(&nbytes) = frame_ + frameIdx_;
            frameIdx_ += size;
        }
        size = calc_binint(nbytes, size);
        if (size < 0) {
            PickleState *st = _Pickle_GetGlobalState();
            /* Corrupt or hostile pickle -- we never write one like this */
            PyErr_SetString(st->UnpicklingError,
                            "LONG pickle has negative byte count");
            return -1;
        }

        if (size == 0)
            value = PyLong_FromLong(0L);
        else {
            bool newFlag = false;
            /* Read the raw little-endian bytes and convert. */
            if (frameLen_ - frameIdx_ < size) {
                pdata = (char *) PyMem_Malloc(size);
                if(pdata == NULL)
                    return -1;
                newFlag = true;
                size_t begIdx=0, actualSize=0;
                const size_t BUFFSIZE = 65536;
                while(begIdx < size) {
                    actualSize = std::min(size - begIdx, BUFFSIZE);
                    if((ret = in_->readBytes(pdata + begIdx, actualSize, actualSize)) != OK)
                        return -1;
                    begIdx += actualSize;
                }
            } else {
                *(&pdata) = frame_ + frameIdx_;
                frameIdx_ += size;
            }
            value = _PyLong_FromByteArray((unsigned char *)pdata, (size_t)size,
                                          1 /* little endian */ , 1 /* signed */ );
            if(newFlag)
                PyMem_Free(pdata);
        }
        if (value == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_float(IO_ERR& ret) {
        PyObject *value;
        char *endptr;
        Py_ssize_t len;
        double d;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();
        errno = 0;
        d = PyOS_string_to_double(tem.c_str(), &endptr, PyExc_OverflowError);
        if (d == -1.0 && PyErr_Occurred())
            return -1;
        if ((endptr[0] != '\n') && (endptr[0] != '\0')) {
            PyErr_SetString(PyExc_ValueError, "could not convert string to float");
            return -1;
        }
        value = PyFloat_FromDouble(d);
        if (value == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_binfloat(IO_ERR& ret) {
        PyObject *value;
        double x;
        char *s;
        if (frameLen_ - frameIdx_ < 8) {
            if((ret = in_->readBytes(shortBuf_, 8, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += 8;
        }
        x = _PyFloat_Unpack8((unsigned char *)s, 0);
        if (x == -1.0 && PyErr_Occurred())
            return -1;
        if ((value = PyFloat_FromDouble(x)) == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_string(IO_ERR& ret) {
        PyObject *bytes;
        PyObject *obj;
        Py_ssize_t len;
        char *p;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        /* Strip the newline */
        len--;
        /* Strip outermost quotes */
        if (len >= 2 && tem[0] == tem[len - 1] && (tem[0] == '\'' || tem[0] == '"')) {
            p = (char*)(tem.c_str()) + 1;
            len -= 2;
        }
        else {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_SetString(st->UnpicklingError,
                            "the STRING opcode argument must be quoted");
            return -1;
        }
        assert(len >= 0);

        /* Use the PyBytes API to decode the string, since that is what is used
           to encode, and then coerce the result to Unicode. */
        bytes = PyBytes_DecodeEscape(p, len, NULL, 0, NULL);
        if (bytes == NULL)
            return -1;
        /* Leave the Python 2.x strings as bytes if the *encoding* given to the
           Unpickler was 'bytes'. Otherwise, convert them to unicode. */
        if (strcmp(unpickler_->encoding, "bytes") == 0) {
            obj = bytes;
        }
        else {
            obj = PyUnicode_FromEncodedObject(bytes, unpickler_->encoding, unpickler_->errors);
            Py_DECREF(bytes);
            if (obj == NULL) {
                return -1;
            }
        }
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }

    int PickleUnmarshall::load_counted_binstring(size_t nbytes, IO_ERR &ret) {
        PyObject *obj;
        char *s;
        if (frameLen_ - frameIdx_ < nbytes) {
            if((ret = in_->readBytes(shortBuf_, nbytes, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += nbytes;
        }
        Py_ssize_t size = calc_binsize(s, nbytes);
        if (size < 0) {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_Format(st->UnpicklingError,
                         "BINSTRING exceeds system's maximum size of %zd bytes",
                         PY_SSIZE_T_MAX);
            return -1;
        }
        bool newFlag = false;
        if (frameLen_ - frameIdx_ < size) {
            s = (char *) PyMem_Malloc(size);
            if (s == NULL)
                return -1;
            newFlag = true;
            size_t begIdx=0, actualSize=0;
            const size_t BUFFSIZE = 65536;
            while(begIdx < size) {
                actualSize = std::min(size - begIdx, BUFFSIZE);
                if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                    return -1;
                begIdx += actualSize;
            }
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += size;
        }
        /* Convert Python 2.x strings to bytes if the *encoding* given to the
           Unpickler was 'bytes'. Otherwise, convert them to unicode. */
        if (strcmp(unpickler_->encoding, "bytes") == 0) {
            obj = PyBytes_FromStringAndSize(s, size);
        }
        else {
            obj = PyUnicode_Decode(s, size, unpickler_->encoding, unpickler_->errors);
        }
        if (newFlag)
            PyMem_Free(s);
        if (obj == NULL) {
            return -1;
        }
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }
    static PyBytesObject *nullstring;
    static PyObject *
    _PyBytes_FromSize(Py_ssize_t size, int use_calloc)
    {
        PyBytesObject *op;
        assert(size >= 0);

        if (size == 0 && (op = nullstring) != NULL) {
#ifdef COUNT_ALLOCS
            null_strings++;
#endif
            Py_INCREF(op);
            return (PyObject *)op;
        }

        if ((size_t)size > (size_t)PY_SSIZE_T_MAX - PyBytesObject_SIZE) {
            PyErr_SetString(PyExc_OverflowError,
                            "byte string is too large");
            return NULL;
        }

        /* Inline PyObject_NewVar */
        if (use_calloc)
            op = (PyBytesObject *)PyObject_Calloc(1, PyBytesObject_SIZE + size);
        else
            op = (PyBytesObject *)PyObject_Malloc(PyBytesObject_SIZE + size);
        if (op == NULL)
            return PyErr_NoMemory();
        (void)PyObject_INIT_VAR(op, &PyBytes_Type, size);
        op->ob_shash = -1;
        if (!use_calloc)
            op->ob_sval[size] = '\0';
        /* empty byte string singleton */
        if (size == 0) {
            nullstring = op;
            Py_INCREF(op);
        }
        return (PyObject *) op;
    }

    int PickleUnmarshall::load_counted_binbytes(size_t nbytes, IO_ERR &ret) {
        PyObject *bytes;
        Py_ssize_t size;
        char *s;
        if (frameLen_ - frameIdx_ < nbytes) {
            if((ret = in_->readBytes(shortBuf_, nbytes, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += nbytes;
        }
        size = calc_binsize(s, nbytes);
        if (size < 0) {
            PyErr_Format(PyExc_OverflowError,
                         "BINBYTES exceeds system's maximum size of %zd bytes",
                         PY_SSIZE_T_MAX);
            return -1;
        }

        if (frameLen_ - frameIdx_ < size) {
            PyBytesObject *op  = (PyBytesObject *)_PyBytes_FromSize(size, 0);
            if (op == NULL)
                return -1;
            size_t begIdx=0, actualSize=0;
            const size_t BUFFSIZE = 65536;
            while(begIdx < size) {
                actualSize = std::min(size - begIdx, BUFFSIZE);
                if((ret = in_->readBytes(op->ob_sval+begIdx, actualSize, actualSize)) != OK)
                    return -1;
                begIdx += actualSize;
            }
            PDATA_PUSH(unpickler_->stack, (PyObject *)op, -1);
            return 0;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += size;
            bytes = PyBytes_FromStringAndSize(s, size);
            if (bytes == NULL)
                return -1;
            PDATA_PUSH(unpickler_->stack, bytes, -1);
            return 0;
        }
    }

    int PickleUnmarshall::load_unicode(IO_ERR &ret) {
        PyObject *str;
        Py_ssize_t len;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 1)
            return bad_readline();
        str = PyUnicode_DecodeRawUnicodeEscape(tem.c_str(), len - 1, NULL);
        if (str == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, str, -1);
        return 0;
    }

    int PickleUnmarshall::load_counted_binunicode(size_t nbytes, IO_ERR &ret) {
        PyObject *str;
        Py_ssize_t size;
        char *s;
        if (frameLen_ - frameIdx_ < nbytes) {
            if((ret = in_->readBytes(shortBuf_, nbytes, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += nbytes;
        }
        size = calc_binsize(s, nbytes);
        if (size < 0) {
            PyErr_Format(PyExc_OverflowError,
                         "BINUNICODE exceeds system's maximum size of %zd bytes",
                         PY_SSIZE_T_MAX);
            return -1;
        }

        bool newFlag = false;
        if (frameLen_ - frameIdx_ < size) {
            s = (char *) PyMem_Malloc(size);
            if (s == NULL)
                return -1;
            newFlag = true;
            size_t begIdx=0, actualSize=0;
            const size_t BUFFSIZE = 65536;
            while(begIdx < size) {
                actualSize = std::min(size - begIdx, BUFFSIZE);
                if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                    return -1;
                begIdx += actualSize;
            }
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += size;
        }
        str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
        if (newFlag)
            PyMem_Free(s);
        if (str == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, str, -1);
        return 0;
    }

    int PickleUnmarshall::load_counted_tuple(Py_ssize_t len) {
        PyObject *tuple;
        if (Py_SIZE(unpickler_->stack) < len)
            return Pdata_stack_underflow(unpickler_->stack);
        tuple = Pdata_poptuple(unpickler_->stack, Py_SIZE(unpickler_->stack) - len);
        if (tuple == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, tuple, -1);
        return 0;
    }

    int PickleUnmarshall::load_tuple() {
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        return load_counted_tuple(Py_SIZE(unpickler_->stack) - i);
    }

    int PickleUnmarshall::load_empty_list() {
        PyObject *list;
        if ((list = PyList_New(0)) == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, list, -1);
        return 0;
    }

    int PickleUnmarshall::load_empty_dict() {
        PyObject *dict;
        if ((dict = PyDict_New()) == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, dict, -1);
        return 0;
    }

    int PickleUnmarshall::load_empty_set() {
        PyObject *set;
        if ((set = PySet_New(NULL)) == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, set, -1);
        return 0;
    }

    int PickleUnmarshall::load_list() {
        PyObject *list;
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        list = Pdata_poplist(unpickler_->stack, i);
        if (list == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, list, -1);
        return 0;
    }

    int PickleUnmarshall::load_dict() {
        PyObject *dict, *key, *value;
        Py_ssize_t i, j, k;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        j = Py_SIZE(unpickler_->stack);
        if ((dict = PyDict_New()) == NULL)
            return -1;
        if ((j - i) % 2 != 0) {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_SetString(st->UnpicklingError, "odd number of items for DICT");
            Py_DECREF(dict);
            return -1;
        }
        for (k = i + 1; k < j; k += 2) {
            key = unpickler_->stack->data[k - 1];
            value = unpickler_->stack->data[k];
            if (PyDict_SetItem(dict, key, value) < 0) {
                Py_DECREF(dict);
                return -1;
            }
        }
        Pdata_clear(unpickler_->stack, i);
        PDATA_PUSH(unpickler_->stack, dict, -1);
        return 0;
    }

    int PickleUnmarshall::load_frozenset() {
        PyObject *items;
        PyObject *frozenset;
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        items = Pdata_poptuple(unpickler_->stack, i);
        if (items == NULL)
            return -1;
        frozenset = PyFrozenSet_New(items);
        Py_DECREF(items);
        if (frozenset == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, frozenset, -1);
        return 0;
    }

    inline static PyObject * instantiate(PyObject *cls, PyObject *args) {
        /* Caller must assure args are a tuple.  Normally, args come from
           Pdata_poptuple which packs objects from the top of the stack
           into a newly created tuple. */
        assert(PyTuple_Check(args));
        if (!PyTuple_GET_SIZE(args) && PyType_Check(cls)) {
            _Py_IDENTIFIER(__getinitargs__);
            _Py_IDENTIFIER(__new__);
            PyObject *func;
            if (_PyObject_LookupAttrId(cls, &PyId___getinitargs__, &func) < 0) {
                return NULL;
            }
            if (func == NULL) {
                return _PyObject_CallMethodIdObjArgs(cls, &PyId___new__, cls, NULL);
            }
            Py_DECREF(func);
        }
        return PyObject_CallObject(cls, args);
    }

    int PickleUnmarshall::load_obj() {
        PyObject *cls, *args, *obj = NULL;
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        if (Py_SIZE(unpickler_->stack) - i < 1)
            return Pdata_stack_underflow(unpickler_->stack);
        args = Pdata_poptuple(unpickler_->stack, i + 1);
        if (args == NULL)
            return -1;
        PDATA_POP(unpickler_->stack, cls);
        if (cls) {
            obj = instantiate(cls, args);
            Py_DECREF(cls);
        }
        Py_DECREF(args);
        if (obj == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }

    int PickleUnmarshall::load_inst(IO_ERR &ret) {
        PyObject *cls = NULL;
        PyObject *args = NULL;
        PyObject *obj = NULL;
        PyObject *module_name;
        PyObject *class_name;
        Py_ssize_t len;
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();

        /* Here it is safe to use PyUnicode_DecodeASCII(), even though non-ASCII
           identifiers are permitted in Python 3.0, since the INST opcode is only
           supported by older protocols on Python 2.x. */
        module_name = PyUnicode_DecodeASCII(tem.c_str(), len - 1, "strict");
        if (module_name == NULL)
            return -1;

        string tem2;
        if ((ret = in_->readLine(tem2)) == OK) {
            len = tem2.size();
            if (len < 2) {
                Py_DECREF(module_name);
                return bad_readline();
            }
            class_name = PyUnicode_DecodeASCII(tem2.c_str(), len - 1, "strict");
            if (class_name != NULL) {
                cls = find_class(unpickler_, module_name, class_name);
                Py_DECREF(class_name);
            }
        }
        Py_DECREF(module_name);
        if (cls == NULL)
            return -1;

        if ((args = Pdata_poptuple(unpickler_->stack, i)) != NULL) {
            obj = instantiate(cls, args);
            Py_DECREF(args);
        }
        Py_DECREF(cls);
        if (obj == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }

    int PickleUnmarshall::load_newobj() {
        PyObject *args = NULL;
        PyObject *clsraw = NULL;
        PyTypeObject *cls;          /* clsraw cast to its true type */
        PyObject *obj;
        PyObject * _pic = PyImport_ImportModule("_pickle");
        PickleState *st = _Pickle_GetState(_pic);
        /* Stack is ... cls argtuple, and we want to call
         * cls.__new__(cls, *argtuple).
         */
        PDATA_POP(unpickler_->stack, args);
        if (args == NULL)
            goto error;
        if (!PyTuple_Check(args)) {
            PyErr_SetString(st->UnpicklingError,
                            "NEWOBJ expected an arg " "tuple.");
            goto error;
        }
        PDATA_POP(unpickler_->stack, clsraw);
        cls = (PyTypeObject *)clsraw;
        if (cls == NULL)
            goto error;
        if (!PyType_Check(cls)) {
            PyErr_SetString(st->UnpicklingError, "NEWOBJ class argument "
                                                 "isn't a type object");
            goto error;
        }
        if (cls->tp_new == NULL) {
            PyErr_SetString(st->UnpicklingError, "NEWOBJ class argument "
                                                 "has NULL tp_new");
            goto error;
        }
        /* Call __new__. */
        obj = cls->tp_new(cls, args, NULL);
        if (obj == NULL)
            goto error;
        Py_DECREF(args);
        Py_DECREF(clsraw);
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;

        error:
        Py_XDECREF(args);
        Py_XDECREF(clsraw);
        return -1;
    }

    int PickleUnmarshall::load_newobj_ex() {
        PyObject *cls, *args, *kwargs;
        PyObject *obj;
        PickleState *st = _Pickle_GetGlobalState();
        PDATA_POP(unpickler_->stack, kwargs);
        if (kwargs == NULL) {
            return -1;
        }
        PDATA_POP(unpickler_->stack, args);
        if (args == NULL) {
            Py_DECREF(kwargs);
            return -1;
        }
        PDATA_POP(unpickler_->stack, cls);
        if (cls == NULL) {
            Py_DECREF(kwargs);
            Py_DECREF(args);
            return -1;
        }
        if (!PyType_Check(cls)) {
            PyErr_Format(st->UnpicklingError,
                         "NEWOBJ_EX class argument must be a type, not %.200s",
                         Py_TYPE(cls)->tp_name);
            goto error;
        }

        if (((PyTypeObject *)cls)->tp_new == NULL) {
            PyErr_SetString(st->UnpicklingError,
                            "NEWOBJ_EX class argument doesn't have __new__");
            goto error;
        }
        if (!PyTuple_Check(args)) {
            PyErr_Format(st->UnpicklingError,
                         "NEWOBJ_EX args argument must be a tuple, not %.200s",
                         Py_TYPE(args)->tp_name);
            goto error;
        }
        if (!PyDict_Check(kwargs)) {
            PyErr_Format(st->UnpicklingError,
                         "NEWOBJ_EX kwargs argument must be a dict, not %.200s",
                         Py_TYPE(kwargs)->tp_name);
            goto error;
        }
        obj = ((PyTypeObject *)cls)->tp_new((PyTypeObject *)cls, args, kwargs);
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(cls);
        if (obj == NULL) {
            return -1;
        }
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;

        error:
        Py_DECREF(kwargs);
        Py_DECREF(args);
        Py_DECREF(cls);
        return -1;
    }

    int PickleUnmarshall::load_global(IO_ERR& ret) {
        PyObject *global = NULL;
        PyObject *module_name;
        PyObject *global_name;
        Py_ssize_t len;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();
        module_name = PyUnicode_DecodeUTF8(tem.c_str(), len - 1, "strict");
        if (!module_name)
            return -1;
        string tem2;
        if ((ret = in_->readLine(tem2)) == OK) {
            len = tem2.size();
            if (len < 2) {
                Py_DECREF(module_name);
                return bad_readline();
            }
            global_name = PyUnicode_DecodeUTF8(tem2.c_str(), len - 1, "strict");
            if (global_name) {
                global = find_class(unpickler_, module_name, global_name);
                Py_DECREF(global_name);
            }
        }
        Py_DECREF(module_name);
        if (global == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, global, -1);
        return 0;
    }

    int PickleUnmarshall::load_stack_global() {
        PyObject *global;
        PyObject *module_name;
        PyObject *global_name;
        PDATA_POP(unpickler_->stack, global_name);
        PDATA_POP(unpickler_->stack, module_name);
        if (module_name == NULL || !PyUnicode_CheckExact(module_name) ||
            global_name == NULL || !PyUnicode_CheckExact(global_name)) {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_SetString(st->UnpicklingError, "STACK_GLOBAL requires str");
            Py_XDECREF(global_name);
            Py_XDECREF(module_name);
            return -1;
        }
        global = find_class(unpickler_, module_name, global_name);
        Py_DECREF(global_name);
        Py_DECREF(module_name);
        if (global == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, global, -1);
        return 0;
    }

    int PickleUnmarshall::load_persid(IO_ERR &ret) {
        PyObject *pid, *obj;
        Py_ssize_t len;
        string tem;
        if (unpickler_->pers_func) {
            if ((ret = in_->readLine(tem)) != OK)
                return -1;
            len = tem.size();
            if (len < 1)
                return bad_readline();
            pid = PyUnicode_DecodeASCII(tem.c_str(), len - 1, "strict");
            if (pid == NULL) {
                if (PyErr_ExceptionMatches(PyExc_UnicodeDecodeError)) {
                    PyErr_SetString(_Pickle_GetGlobalState()->UnpicklingError,
                                    "persistent IDs in protocol 0 must be "
                                    "ASCII strings");
                }
                return -1;
            }
            obj = call_method(unpickler_->pers_func, unpickler_->pers_func_self, pid);
            Py_DECREF(pid);
            if (obj == NULL)
                return -1;
            PDATA_PUSH(unpickler_->stack, obj, -1);
            return 0;
        }
        else {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_SetString(st->UnpicklingError,
                            "A load persistent id instruction was encountered,\n"
                            "but no persistent_load function was specified.");
            return -1;
        }
    }

    int PickleUnmarshall::load_binpersid() {
        PyObject *pid, *obj;
        if (unpickler_->pers_func) {
            PDATA_POP(unpickler_->stack, pid);
            if (pid == NULL)
                return -1;

            obj = call_method(unpickler_->pers_func, unpickler_->pers_func_self, pid);
            Py_DECREF(pid);
            if (obj == NULL)
                return -1;
            PDATA_PUSH(unpickler_->stack, obj, -1);
            return 0;
        }
        else {
            PickleState *st = _Pickle_GetGlobalState();
            PyErr_SetString(st->UnpicklingError,
                            "A load persistent id instruction was encountered,\n"
                            "but no persistent_load function was specified.");
            return -1;
        }
    }

    int PickleUnmarshall::load_pop() {
        Py_ssize_t len = Py_SIZE(unpickler_->stack);

        /* Note that we split the (pickle.py) stack into two stacks,
         * an object stack and a mark stack. We have to be clever and
         * pop the right one. We do this by looking at the top of the
         * mark stack first, and only signalling a stack underflow if
         * the object stack is empty and the mark stack doesn't match
         * our expectations.
         */
        if (unpickler_->num_marks > 0 && unpickler_->marks[unpickler_->num_marks - 1] == len) {
            unpickler_->num_marks--;
            unpickler_->stack->mark_set = unpickler_->num_marks != 0;
            unpickler_->stack->fence = unpickler_->num_marks ? unpickler_->marks[unpickler_->num_marks - 1] : 0;
        } else if (len <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        else {
            len--;
            Py_DECREF(unpickler_->stack->data[len]);
            Py_SIZE(unpickler_->stack) = len;
        }
        return 0;
    }

    int PickleUnmarshall::load_pop_mark() {
        Py_ssize_t i;
        if ((i = marker(unpickler_)) < 0)
            return -1;
        Pdata_clear(unpickler_->stack, i);
        return 0;
    }

    int PickleUnmarshall::load_dup() {
        PyObject *last;
        Py_ssize_t len = Py_SIZE(unpickler_->stack);

        if (len <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        last = unpickler_->stack->data[len - 1];
        PDATA_APPEND(unpickler_->stack, last, -1);
        return 0;
    }

    int PickleUnmarshall::load_get(IO_ERR& ret) {
        PyObject *key, *value;
        Py_ssize_t idx;
        Py_ssize_t len;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();
        key = PyLong_FromString(tem.c_str(), NULL, 10);
        if (key == NULL)
            return -1;
        idx = PyLong_AsSsize_t(key);
        if (idx == -1 && PyErr_Occurred()) {
            Py_DECREF(key);
            return -1;
        }
        value = _Unpickler_MemoGet(unpickler_, idx);
        if (value == NULL) {
            if (!PyErr_Occurred())
                PyErr_SetObject(PyExc_KeyError, key);
            Py_DECREF(key);
            return -1;
        }
        Py_DECREF(key);
        PDATA_APPEND(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_binget(IO_ERR &ret) {
        PyObject *value;
        Py_ssize_t idx;
        char *s;
        if (frameLen_ - frameIdx_ < 1) {
            if((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += 1;
        }
        idx = Py_CHARMASK(s[0]);
        value = _Unpickler_MemoGet(unpickler_, idx);
        if (value == NULL) {
            PyObject *key = PyLong_FromSsize_t(idx);
            if (key != NULL) {
                PyErr_SetObject(PyExc_KeyError, key);
                Py_DECREF(key);
            }
            return -1;
        }
        PDATA_APPEND(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_long_binget(IO_ERR &ret) {
        PyObject *value;
        Py_ssize_t idx;
        char *s;
        if (frameLen_ - frameIdx_ < 4) {
            if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += 4;
        }
        idx = calc_binsize(s, 4);
        value = _Unpickler_MemoGet(unpickler_, idx);
        if (value == NULL) {
            PyObject *key = PyLong_FromSsize_t(idx);
            if (key != NULL) {
                PyErr_SetObject(PyExc_KeyError, key);
                Py_DECREF(key);
            }
            return -1;
        }
        PDATA_APPEND(unpickler_->stack, value, -1);
        return 0;
    }

    int PickleUnmarshall::load_extension(size_t nbytes, IO_ERR& ret) {
        char *codebytes;            /* the nbytes bytes after the opcode */
        long code;                  /* calc_binint returns long */
        PyObject *py_code;          /* code as a Python int */
        PyObject *obj;              /* the object to push */
        PyObject *pair;             /* (module_name, class_name) */
        PyObject *module_name, *class_name;
        PickleState *st = _Pickle_GetGlobalState();

        assert(nbytes == 1 || nbytes == 2 || nbytes == 4);
        if (frameLen_ - frameIdx_ < nbytes) {
            if((ret = in_->readBytes(shortBuf_, nbytes, false)) != OK)
                return -1;
            *(&codebytes) = shortBuf_;
        } else {
            *(&codebytes) = frame_ + frameIdx_;
            frameIdx_ += nbytes;
        }
        code = calc_binint(codebytes, nbytes);
        if (code <= 0) {            /* note that 0 is forbidden */
            /* Corrupt or hostile pickle. */
            PyErr_SetString(st->UnpicklingError, "EXT specifies code <= 0");
            return -1;
        }
        /* Look for the code in the cache. */
        py_code = PyLong_FromLong(code);
        if (py_code == NULL)
            return -1;
        obj = PyDict_GetItemWithError(st->extension_cache, py_code);
        if (obj != NULL) {
            /* Bingo. */
            Py_DECREF(py_code);
            PDATA_APPEND(unpickler_->stack, obj, -1);
            return 0;
        }
        if (PyErr_Occurred()) {
            Py_DECREF(py_code);
            return -1;
        }
        /* Look up the (module_name, class_name) pair. */
        pair = PyDict_GetItemWithError(st->inverted_registry, py_code);
        if (pair == NULL) {
            Py_DECREF(py_code);
            if (!PyErr_Occurred()) {
                PyErr_Format(PyExc_ValueError, "unregistered extension "
                                               "code %ld", code);
            }
            return -1;
        }
        /* Since the extension registry is manipulable via Python code,
         * confirm that pair is really a 2-tuple of strings.
         */
        if (!PyTuple_Check(pair) || PyTuple_Size(pair) != 2 ||
            !PyUnicode_Check(module_name = PyTuple_GET_ITEM(pair, 0)) ||
            !PyUnicode_Check(class_name = PyTuple_GET_ITEM(pair, 1))) {
            Py_DECREF(py_code);
            PyErr_Format(PyExc_ValueError, "_inverted_registry[%ld] "
                                           "isn't a 2-tuple of strings", code);
            return -1;
        }
        /* Load the object. */
        obj = find_class(unpickler_, module_name, class_name);
        if (obj == NULL) {
            Py_DECREF(py_code);
            return -1;
        }
        /* Cache code -> obj. */
        code = PyDict_SetItem(st->extension_cache, py_code, obj);
        Py_DECREF(py_code);
        if (code < 0) {
            Py_DECREF(obj);
            return -1;
        }
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }

    int PickleUnmarshall::load_put(IO_ERR &ret) {
        PyObject *key, *value;
        Py_ssize_t idx;
        Py_ssize_t len;
        string tem;
        if ((ret = in_->readLine(tem)) != OK)
            return -1;
        len = tem.size();
        if (len < 2)
            return bad_readline();

        if (Py_SIZE(unpickler_->stack) <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        value = unpickler_->stack->data[Py_SIZE(unpickler_->stack) - 1];
        key = PyLong_FromString(tem.c_str(), NULL, 10);
        if (key == NULL)
            return -1;
        idx = PyLong_AsSsize_t(key);
        Py_DECREF(key);
        if (idx < 0) {
            if (!PyErr_Occurred())
                PyErr_SetString(PyExc_ValueError,
                                "negative PUT argument");
            return -1;
        }
        return _Unpickler_MemoPut(unpickler_, idx, value);
    }

    int PickleUnmarshall::load_binput(IO_ERR& ret) {
        PyObject *value;
        Py_ssize_t idx;
        char *s;
        if (frameLen_ - frameIdx_ < 1) {
            if((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += 1;
        }
        if (Py_SIZE(unpickler_->stack) <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        value = unpickler_->stack->data[Py_SIZE(unpickler_->stack) - 1];
        idx = Py_CHARMASK(s[0]);
        return _Unpickler_MemoPut(unpickler_, idx, value);
    }

    int PickleUnmarshall::load_long_binput(IO_ERR &ret) {
        PyObject *value;
        Py_ssize_t idx;
        char *s;
        if (frameLen_ - frameIdx_ < 4) {
            if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                return -1;
            *(&s) = shortBuf_;
        } else {
            *(&s) = frame_ + frameIdx_;
            frameIdx_ += 4;
        }
        if (Py_SIZE(unpickler_->stack) <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        value = unpickler_->stack->data[Py_SIZE(unpickler_->stack) - 1];
        idx = calc_binsize(s, 4);
        if (idx < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "negative LONG_BINPUT argument");
            return -1;
        }
        return _Unpickler_MemoPut(unpickler_, idx, value);
    }

    int PickleUnmarshall::load_memoize() {
        PyObject *value;
        if (Py_SIZE(unpickler_->stack) <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        value = unpickler_->stack->data[Py_SIZE(unpickler_->stack) - 1];
        return _Unpickler_MemoPut(unpickler_, unpickler_->memo_len, value);
    }

    static int do_append(UnpicklerObject *self, Py_ssize_t x) {

        PyObject *value;
        PyObject *slice;
        PyObject *list;
        PyObject *result;
        Py_ssize_t len, i;
        len = Py_SIZE(self->stack);
        if (x > len || x <= self->stack->fence)
            return Pdata_stack_underflow(self->stack);
        if (len == x)  /* nothing to do */
            return 0;
        list = self->stack->data[x - 1];
        if (PyList_CheckExact(list)) {
            Py_ssize_t list_len;
            int ret;

            slice = Pdata_poplist(self->stack, x);
            if (!slice)
                return -1;
            list_len = PyList_GET_SIZE(list);
            ret = PyList_SetSlice(list, list_len, list_len, slice);
            Py_DECREF(slice);
            return ret;
        }
        else {
            PyObject *extend_func;
            _Py_IDENTIFIER(extend);
            extend_func = _PyObject_GetAttrId(list, &PyId_extend);
            if (extend_func != NULL) {
                slice = Pdata_poplist(self->stack, x);
                if (!slice) {
                    Py_DECREF(extend_func);
                    return -1;
                }
                result = _Pickle_FastCall(extend_func, slice);
                Py_DECREF(extend_func);
                if (result == NULL)
                    return -1;
                Py_DECREF(result);
            }
            else {
                PyObject *append_func;
                _Py_IDENTIFIER(append);
                /* Even if the PEP 307 requires extend() and append() methods,
                   fall back on append() if the object has no extend() method
                   for backward compatibility. */
                PyErr_Clear();
                append_func = _PyObject_GetAttrId(list, &PyId_append);
                if (append_func == NULL)
                    return -1;
                for (i = x; i < len; i++) {
                    value = self->stack->data[i];
                    result = _Pickle_FastCall(append_func, value);
                    if (result == NULL) {
                        Pdata_clear(self->stack, i + 1);
                        Py_SIZE(self->stack) = x;
                        Py_DECREF(append_func);
                        return -1;
                    }
                    Py_DECREF(result);
                }
                Py_SIZE(self->stack) = x;
                Py_DECREF(append_func);
            }
        }
        return 0;
    }

    int PickleUnmarshall::load_append() {
        if (Py_SIZE(unpickler_->stack) - 1 <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        return do_append(unpickler_, Py_SIZE(unpickler_->stack) - 1);
    }

    int PickleUnmarshall::load_appends() {
        Py_ssize_t i = marker(unpickler_);
        if (i < 0)
            return -1;
        return do_append(unpickler_, i);
    }

    static int do_setitems(UnpicklerObject *self, Py_ssize_t x) {
        PyObject *value, *key;
        PyObject *dict;
        Py_ssize_t len, i;
        int status = 0;
        len = Py_SIZE(self->stack);
        if (x > len || x <= self->stack->fence)
            return Pdata_stack_underflow(self->stack);
        if (len == x)  /* nothing to do */
            return 0;
        if ((len - x) % 2 != 0) {
            PickleState *st = _Pickle_GetGlobalState();
            /* Currupt or hostile pickle -- we never write one like this. */
            PyErr_SetString(st->UnpicklingError,
                            "odd number of items for SETITEMS");
            return -1;
        }
        /* Here, dict does not actually need to be a PyDict; it could be anything
           that supports the __setitem__ attribute. */
        dict = self->stack->data[x - 1];
        for (i = x + 1; i < len; i += 2) {
            key = self->stack->data[i - 1];
            value = self->stack->data[i];
            if (PyObject_SetItem(dict, key, value) < 0) {
                status = -1;
                break;
            }
        }
        Pdata_clear(self->stack, x);
        return status;
    }

    int PickleUnmarshall::load_setitem() {
        return do_setitems(unpickler_, Py_SIZE(unpickler_->stack) - 2);
    }

    int PickleUnmarshall::load_setitems() {
        Py_ssize_t i = marker(unpickler_);
        if (i < 0)
            return -1;
        return do_setitems(unpickler_, i);
    }

    int PickleUnmarshall::load_additems() {
        PyObject *set;
        Py_ssize_t mark, len, i;
        mark =  marker(unpickler_);
        if (mark < 0)
            return -1;
        len = Py_SIZE(unpickler_->stack);
        if (mark > len || mark <= unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        if (len == mark)  /* nothing to do */
            return 0;
        set = unpickler_->stack->data[mark - 1];
        if (PySet_Check(set)) {
            PyObject *items;
            int status;
            items = Pdata_poptuple(unpickler_->stack, mark);
            if (items == NULL)
                return -1;
            status = _PySet_Update(set, items);
            Py_DECREF(items);
            return status;
        }
        else {
            PyObject *add_func;
            _Py_IDENTIFIER(add);
            add_func = _PyObject_GetAttrId(set, &PyId_add);
            if (add_func == NULL)
                return -1;
            for (i = mark; i < len; i++) {
                PyObject *result;
                PyObject *item;
                item = unpickler_->stack->data[i];
                result = _Pickle_FastCall(add_func, item);
                if (result == NULL) {
                    Pdata_clear(unpickler_->stack, i + 1);
                    Py_SIZE(unpickler_->stack) = mark;
                    return -1;
                }
                Py_DECREF(result);
            }
            Py_SIZE(unpickler_->stack) = mark;
        }
        return 0;
    }

    int PickleUnmarshall::load_build() {
        PyObject *state, *inst, *slotstate;
        PyObject *setstate;
        int status = 0;
        _Py_IDENTIFIER(__setstate__);
        /* Stack is ... instance, state.  We want to leave instance at
         * the stack top, possibly mutated via instance.__setstate__(state).
         */
        if (Py_SIZE(unpickler_->stack) - 2 < unpickler_->stack->fence)
            return Pdata_stack_underflow(unpickler_->stack);
        PDATA_POP(unpickler_->stack, state);
        if (state == NULL)
            return -1;
        inst = unpickler_->stack->data[Py_SIZE(unpickler_->stack) - 1];
        if (_PyObject_LookupAttrId(inst, &PyId___setstate__, &setstate) < 0) {
            Py_DECREF(state);
            return -1;
        }
        if (setstate != NULL) {
            PyObject *result;
            /* The explicit __setstate__ is responsible for everything. */
            result = _Pickle_FastCall(setstate, state);
            Py_DECREF(setstate);
            if (result == NULL)
                return -1;
            Py_DECREF(result);
            return 0;
        }
        /* A default __setstate__.  First see whether state embeds a
         * slot state dict too (a proto 2 addition).
         */
        if (PyTuple_Check(state) && PyTuple_GET_SIZE(state) == 2) {
            PyObject *tmp = state;
            state = PyTuple_GET_ITEM(tmp, 0);
            slotstate = PyTuple_GET_ITEM(tmp, 1);
            Py_INCREF(state);
            Py_INCREF(slotstate);
            Py_DECREF(tmp);
        }
        else
            slotstate = NULL;
        /* Set inst.__dict__ from the state dict (if any). */
        if (state != Py_None) {
            PyObject *dict;
            PyObject *d_key, *d_value;
            Py_ssize_t i;
            _Py_IDENTIFIER(__dict__);
            if (!PyDict_Check(state)) {
                PickleState *st = _Pickle_GetGlobalState();
                PyErr_SetString(st->UnpicklingError, "state is not a dictionary");
                goto error;
            }
            dict = _PyObject_GetAttrId(inst, &PyId___dict__);
            if (dict == NULL)
                goto error;
            i = 0;
            while (PyDict_Next(state, &i, &d_key, &d_value)) {
                /* normally the keys for instance attributes are
                   interned.  we should try to do that here. */
                Py_INCREF(d_key);
                if (PyUnicode_CheckExact(d_key))
                    PyUnicode_InternInPlace(&d_key);
                if (PyObject_SetItem(dict, d_key, d_value) < 0) {
                    Py_DECREF(d_key);
                    goto error;
                }
                Py_DECREF(d_key);
            }
            Py_DECREF(dict);
        }
        /* Also set instance attributes from the slotstate dict (if any). */
        if (slotstate != NULL) {
            PyObject *d_key, *d_value;
            Py_ssize_t i;
            if (!PyDict_Check(slotstate)) {
                PickleState *st = _Pickle_GetGlobalState();
                PyErr_SetString(st->UnpicklingError,
                                "slot state is not a dictionary");
                goto error;
            }
            i = 0;
            while (PyDict_Next(slotstate, &i, &d_key, &d_value)) {
                if (PyObject_SetAttr(inst, d_key, d_value) < 0)
                    goto error;
            }
        }

        if (0) {
            error:
            status = -1;
        }
        Py_DECREF(state);
        Py_XDECREF(slotstate);
        return status;
    }

    int PickleUnmarshall::load_mark() {
        if ((unpickler_->num_marks + 1) >= unpickler_->marks_size) {
            size_t alloc;
            /* Use the size_t type to check for overflow. */
            alloc = ((size_t)unpickler_->num_marks << 1) + 20;
            if (alloc > (PY_SSIZE_T_MAX / sizeof(Py_ssize_t)) ||
                alloc <= ((size_t)unpickler_->num_marks + 1)) {
                PyErr_NoMemory();
                return -1;
            }

            Py_ssize_t *marks_old = unpickler_->marks;
            PyMem_RESIZE(unpickler_->marks, Py_ssize_t, alloc);
            if (unpickler_->marks == NULL) {
                PyMem_FREE(marks_old);
                unpickler_->marks_size = 0;
                PyErr_NoMemory();
                return -1;
            }
            unpickler_->marks_size = (Py_ssize_t)alloc;
        }
        unpickler_->stack->mark_set = 1;
        unpickler_->marks[unpickler_->num_marks++] = unpickler_->stack->fence = Py_SIZE(unpickler_->stack);
        return 0;
    }

    int PickleUnmarshall::load_reduce() {
        PyObject *callable = NULL;
        PyObject *argtup = NULL;
        PyObject *obj = NULL;
        PDATA_POP(unpickler_->stack, argtup);
        if (argtup == NULL)
            return -1;
        PDATA_POP(unpickler_->stack, callable);
        if (callable) {
            obj = PyObject_CallObject(callable, argtup);
            Py_DECREF(callable);
        }
        Py_DECREF(argtup);

        if (obj == NULL)
            return -1;
        PDATA_PUSH(unpickler_->stack, obj, -1);
        return 0;
    }

    int PickleUnmarshall::load_proto(IO_ERR &ret) {
        if ((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
            return -1;
        int i = (unsigned char)shortBuf_[0];
        if (i <= HIGHEST_PROTOCOL) {
            unpickler_->proto = i;
            return 0;
        }
        PyErr_Format(PyExc_ValueError, "unsupported pickle protocol: %d", i);
        return -1;
    }

    int PickleUnmarshall::load_frame(IO_ERR &ret) {
        if ((ret = in_->readBytes(shortBuf_, 8, false)) != OK)
            return -1;

        Py_ssize_t frame_len;
        frame_len = calc_binsize(shortBuf_, 8);
        if (frame_len < 0) {
            return -1;
        }
        if(frame_len > frameLen_){
            if(frame_) {
                delete [] frame_;
            }
            frame_ = new char[frame_len];
        }
        size_t begIdx=0, actualSize=0;
        const size_t BUFFSIZE = 4096;
        while(begIdx < frame_len) {
            actualSize = std::min(frame_len - begIdx, BUFFSIZE);
            if((ret = in_->readBytes(frame_ + begIdx, actualSize, actualSize)) != OK)
                return -1;
            begIdx += actualSize;
        }
        frameLen_ = frame_len;
        frameIdx_ = 0;
        return 0;
    }

    static int
    list_resize(PyListObject *self, Py_ssize_t newsize)
    {
        PyObject **items;
        size_t new_allocated, num_allocated_bytes;
        Py_ssize_t allocated = self->allocated;

        /* Bypass realloc() when a previous overallocation is large enough
           to accommodate the newsize.  If the newsize falls lower than half
           the allocated size, then proceed with the realloc() to shrink the list.
        */
        if (allocated >= newsize && newsize >= (allocated >> 1)) {
            assert(self->ob_item != NULL || newsize == 0);
            Py_SIZE(self) = newsize;
            return 0;
        }

        /* This over-allocates proportional to the list size, making room
         * for additional growth.  The over-allocation is mild, but is
         * enough to give linear-time amortized behavior over a long
         * sequence of appends() in the presence of a poorly-performing
         * system realloc().
         * The growth pattern is:  0, 4, 8, 16, 25, 35, 46, 58, 72, 88, ...
         * Note: new_allocated won't overflow because the largest possible value
         *       is PY_SSIZE_T_MAX * (9 / 8) + 6 which always fits in a size_t.
         */
        new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6);
        if (new_allocated > (size_t)PY_SSIZE_T_MAX / sizeof(PyObject *)) {
            PyErr_NoMemory();
            return -1;
        }

        if (newsize == 0)
            new_allocated = 0;
        num_allocated_bytes = new_allocated * sizeof(PyObject *);
        items = (PyObject **)PyMem_Realloc(self->ob_item, num_allocated_bytes);
        if (items == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        self->ob_item = items;
        Py_SIZE(self) = newsize;
        self->allocated = new_allocated;
        return 0;
    }

    int PickleUnmarshall::load_symbol(IO_ERR &ret) {
        char op;
        std::vector<PyObject*> buff;
        if((ret = in_->readChar(op)) != OK)
            return -1;
        while(op == Pickle::opcode::FRAME) {
            if(load_frame(ret) < 0)
                return -1;

            while(frameIdx_ < frameLen_) {
                op = frame_[frameIdx_];
                frameIdx_++;
                switch ((enum Pickle::opcode)op){
                    case Pickle::opcode::SHORT_BINUNICODE: {
                        PyObject *str;
                        Py_ssize_t size;
                        char *s;
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 1;
                        size = calc_binsize(s, 1);
                        if (size < 0) {
                            PyErr_Format(PyExc_OverflowError,
                                         "BINUNICODE exceeds system's maximum size of %zd bytes",
                                         PY_SSIZE_T_MAX);
                            return -1;
                        }
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;
                        str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                        if (str == NULL)
                            return -1;
                        buff.emplace_back(str);
                        break;
                    }
                    case Pickle::opcode::BINUNICODE: {
                        PyObject *str;
                        Py_ssize_t size;
                        char *s;
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 1;
                        size = calc_binsize(s, 4);
                        if (size < 0) {
                            PyErr_Format(PyExc_OverflowError,
                                         "BINUNICODE exceeds system's maximum size of %zd bytes",
                                         PY_SSIZE_T_MAX);
                            return -1;
                        }
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;

                        str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                        if (str == NULL)
                            return -1;
                        buff.emplace_back(str);
                        break;
                    }
                    case Pickle::opcode::BINUNICODE8: {
                        PyObject *str;
                        Py_ssize_t size;
                        char *s;
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 1;
                        size = calc_binsize(s, 8);
                        if (size < 0) {
                            PyErr_Format(PyExc_OverflowError,
                                         "BINUNICODE exceeds system's maximum size of %zd bytes",
                                         PY_SSIZE_T_MAX);
                            return -1;
                        }
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;
                        str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                        if (str == NULL)
                            return -1;

                        buff.emplace_back(str);
                        break;
                    }
                    default:
                        return -1;
                }
            }
            if((ret = in_->readChar(op)) != OK)
                return -1;
        }
        Py_ssize_t size;
        switch ((enum Pickle::opcode)op) {
            case Pickle::opcode::BINBYTES: {
                if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                    return -1;
                size = calc_binsize(shortBuf_, 4);
                break;
            }
            case Pickle::opcode::BINBYTES8: {
                if((ret = in_->readBytes(shortBuf_, 8, false)) != OK)
                    return -1;
                size = calc_binsize(shortBuf_, 8);
                break;
            }
            default:
                return -1;
        }
        char *s;
        if (size < 0) {
            PyErr_Format(PyExc_OverflowError,
                         "BINBYTES exceeds system's maximum size of %zd bytes",
                         PY_SSIZE_T_MAX);
            return -1;
        }
        s = (char *) PyMem_Malloc(size);
        if (s == NULL)
            return -1;
        size_t begIdx=0, actualSize=0;
        const size_t BUFFSIZE = 65536;
        while(begIdx < size) {
            actualSize = std::min(size - begIdx, BUFFSIZE);
            if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                return -1;
            begIdx += actualSize;
        }

        PyObject *value;
        Py_ssize_t idx;
        int cnt = 1;
        for(size_t i = 0; i < size; i = i + 4) {
            idx = calc_binsize(s + i, 4);
            value = buff[idx];
            if (value == NULL) {
                PyObject *key = PyLong_FromSsize_t(idx);
                if (key != NULL) {
                    PyErr_SetObject(PyExc_KeyError, key);
                    Py_DECREF(key);
                }
                PyMem_Free(s);
                return -1;
            }
            PDATA_APPEND(unpickler_->stack, value, -1);
            cnt++;
        }

        PyMem_Free(s);
        return 0;
    }

    int PickleUnmarshall::load_objectBegin(IO_ERR &ret) {
        long col = 0, row = 0;
        char op;
        int cnt = 0;
        int rollbacksize = 0;
        for(int i = 0; i < 4; ++i){
            if(frameIdx_ < frameLen_ ){
                op = frame_[frameIdx_];
                frameIdx_++;
            }else{
                if((ret = in_->readChar(op)) != OK)
                    return -1;
            }
            switch (op) {
                case Pickle::opcode::MARK: if(load_mark() < 0) return -1; continue;
                case Pickle::opcode::BININT: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 4;
                    rollbacksize += 4;
                    PyObject *value;
                    long x = calc_binint(s, 4);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    if (cnt == 1) {
                        col = x;
                    }
                    if (cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    break;
                }
                case Pickle::opcode::BININT1: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 1;
                    rollbacksize += 1;
                    PyObject *value;
                    long x = calc_binint(s, 1);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    if (cnt == 1) {
                        col = x;
                    }
                    if (cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    break;
                }
                case Pickle::opcode::BININT2: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 2;
                    rollbacksize += 2;
                    PyObject *value;
                    long x = calc_binint(s, 2);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    if (cnt == 1) {
                        col = x;
                    }
                    if (cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    break;
                }
                default:
                    return -1;
            }
        }
        int listsize = col * row;
        frameIdx_ -= (rollbacksize+3);
        if (listsize == 0) {
            do {
                if(frameIdx_ < frameLen_ ){
                    op = frame_[frameIdx_];
                    frameIdx_++;
                }else{
                    if((ret = in_->readChar(op)) != OK)
                        return -1;
                }
                switch (op) {
                    case Pickle::opcode::MARK: if(load_mark() < 0) return -1; continue;
                    case Pickle::opcode::BININT: if(load_binint(ret) < 0) return -1; continue;
                    case Pickle::opcode::BININT1: if(load_binint1(ret) < 0) return -1; continue;
                    case Pickle::opcode::BININT2: if(load_binint2(ret) < 0) return -1; continue;
                    case Pickle::opcode::TUPLE1: if(load_counted_tuple(1) < 0) return -1; continue;
                    case Pickle::opcode::TUPLE2: if(load_counted_tuple(2) < 0) return -1; continue;
                    case Pickle::opcode::TUPLE3: if(load_counted_tuple(3) < 0) return -1; continue;
                    case Pickle::opcode::MEMOIZE: if(load_memoize() < 0) return -1; continue;
                    case Pickle::opcode::NEWTRUE: if(load_bool(Py_True) < 0) return -1; continue;
                    case Pickle::opcode::NEWFALSE: if(load_bool(Py_False) < 0) return -1; continue;
                    case Pickle::opcode::BINGET: if(load_binget(ret) < 0) return -1; continue;
                    case Pickle::opcode::EMPTY_LIST: if(load_empty_list() < 0) return -1; continue;
                    case Pickle::opcode::FRAME: break;
                    case Pickle::opcode::SYMBOL: break;
                    default:
                        return -1;
                }
            }while( op != Pickle::opcode::SYMBOL && op != Pickle::opcode::FRAME);

            if(op == Pickle::opcode::FRAME){
                if(load_frame(ret) < 0)
                    return -1;
            }
            return 0;
        }
        else if (listsize == 1) {
            return 0;
        }
        do {
            if(frameIdx_ < frameLen_ ){
                op = frame_[frameIdx_];
                frameIdx_++;
            }else{
                if((ret = in_->readChar(op)) != OK)
                    return -1;
            }
            switch (op) {
                case Pickle::opcode::MARK: if(load_mark() < 0) return -1; continue;
                case Pickle::opcode::BININT: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 4;
                    PyObject *value;
                    long x = calc_binint(s, 4);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    PDATA_PUSH(unpickler_->stack, value, -1);
                    if(cnt == 1) {
                        col = x;
                    }
                    if(cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    continue;
                }
                case Pickle::opcode::BININT1: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 1;
                    PyObject *value;
                    long x = calc_binint(s, 1);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    PDATA_PUSH(unpickler_->stack, value, -1);
                    if(cnt == 1) {
                        col = x;
                    }
                    if(cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    continue;
                }
                case Pickle::opcode::BININT2: {
                    char *s;
                    *(&s) = frame_ + frameIdx_;
                    frameIdx_ += 2;
                    PyObject *value;
                    long x = calc_binint(s, 2);
                    if ((value = PyLong_FromLong(x)) == NULL)
                        return -1;
                    PDATA_PUSH(unpickler_->stack, value, -1);
                    if(cnt == 1) {
                        col = x;
                    }
                    if(cnt == 2) {
                        row = x;
                    }
                    ++cnt;
                    continue;
                }
                case Pickle::opcode::TUPLE1: if(load_counted_tuple(1) < 0) return -1; continue;
                case Pickle::opcode::TUPLE2: if(load_counted_tuple(2) < 0) return -1; continue;
                case Pickle::opcode::TUPLE3: if(load_counted_tuple(3) < 0) return -1; continue;
                case Pickle::opcode::MEMOIZE: if(load_memoize() < 0) return -1; continue;
                case Pickle::opcode::NEWTRUE: if(load_bool(Py_True) < 0) return -1; continue;
                case Pickle::opcode::NEWFALSE: if(load_bool(Py_False) < 0) return -1; continue;
                case Pickle::opcode::BINGET: if(load_binget(ret) < 0) return -1; continue;
                case Pickle::opcode::EMPTY_LIST: if(load_empty_list() < 0) return -1; continue;
                case Pickle::opcode::FRAME: break;
                case Pickle::opcode::SYMBOL: break;
                default:
                    return -1;
            }
        }while( op != Pickle::opcode::SYMBOL && op != Pickle::opcode::FRAME);
        Py_ssize_t x = marker(unpickler_);

        if (x < 0)
            return -1;
            
        PyObject *list;
        list = unpickler_->stack->data[x - 1];

        int count = 0 ;
        int newObjCnt = 0;

        if (listsize > 0) {
            if (list_resize((PyListObject *) list, listsize) < 0)
                return -1;
        }
        std::unordered_map<int, int> map;
        while(op != Pickle::opcode::APPENDS) {
            switch (op) {
                case Pickle::opcode::SYMBOL: {
                    char op1;
                    std::vector<PyObject*> buff;
                    if((ret = in_->readChar(op1)) != OK)
                        return -1;
                    while(op1 == Pickle::opcode::FRAME) {
                        if(load_frame(ret) < 0)
                            return -1;

                        while(frameIdx_ < frameLen_) {
                            op1 = frame_[frameIdx_];
                            frameIdx_++;
                            switch ((enum Pickle::opcode)op1){
                                case Pickle::opcode::SHORT_BINUNICODE: {
                                    PyObject *str;
                                    Py_ssize_t size;
                                    char *s;
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += 1;
                                    size = calc_binsize(s, 1);
                                    if (size < 0) {
                                        PyErr_Format(PyExc_OverflowError,
                                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                                     PY_SSIZE_T_MAX);
                                        return -1;
                                    }
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += size;
                                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                                    if (str == NULL)
                                        return -1;
                                    buff.emplace_back(str);
                                    break;
                                }
                                case Pickle::opcode::BINUNICODE: {
                                    PyObject *str;
                                    Py_ssize_t size;
                                    char *s;
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += 1;
                                    size = calc_binsize(s, 4);
                                    if (size < 0) {
                                        PyErr_Format(PyExc_OverflowError,
                                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                                     PY_SSIZE_T_MAX);
                                        return -1;
                                    }
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += size;

                                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                                    if (str == NULL)
                                        return -1;
                                    buff.emplace_back(str);
                                    break;
                                }
                                case Pickle::opcode::BINUNICODE8: {
                                    PyObject *str;
                                    Py_ssize_t size;
                                    char *s;
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += 1;
                                    size = calc_binsize(s, 8);
                                    if (size < 0) {
                                        PyErr_Format(PyExc_OverflowError,
                                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                                     PY_SSIZE_T_MAX);
                                        return -1;
                                    }
                                    *(&s) = frame_ + frameIdx_;
                                    frameIdx_ += size;
                                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                                    if (str == NULL)
                                        return -1;

                                    buff.emplace_back(str);
                                    break;
                                }
                                default:
                                    return -1;
                            }
                        }
                        if((ret = in_->readChar(op1)) != OK)
                            return -1;
                    }
                    Py_ssize_t size;
                    switch ((enum Pickle::opcode)op1) {
                        case Pickle::opcode::BINBYTES: {
                            if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                                return -1;
                            size = calc_binsize(shortBuf_, 4);
                            break;
                        }
                        case Pickle::opcode::BINBYTES8: {
                            if((ret = in_->readBytes(shortBuf_, 8, false)) != OK)
                                return -1;
                            size = calc_binsize(shortBuf_, 8);
                            break;
                        }
                        default:
                            return -1;
                    }
                    char *s;
                    if (size < 0) {
                        PyErr_Format(PyExc_OverflowError,
                                     "BINBYTES exceeds system's maximum size of %zd bytes",
                                     PY_SSIZE_T_MAX);
                        return -1;
                    }

                    s = (char *) PyMem_Malloc(size);
                    if (s == NULL)
                        return -1;
                    size_t begIdx=0, actualSize=0;

                    const size_t BUFFSIZE = 65536;
                    while(begIdx < size) {
                        actualSize = std::min(size - begIdx, BUFFSIZE);
                        if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                            return -1;
                        begIdx += actualSize;
                    }

                    PyObject *value;
                    Py_ssize_t idx;
                    for(size_t i = 0; i < size; i = i + 4) {
                        idx = calc_binsize(s + i, 4);
                        value = buff[idx];
                        if (value == NULL) {
                            PyObject *key = PyLong_FromSsize_t(idx);
                            if (key != NULL) {
                                PyErr_SetObject(PyExc_KeyError, key);
                                Py_DECREF(key);
                            }
                            PyMem_Free(s);
                            return -1;
                        }

                        Py_IncRef(value);
                        PyList_SET_ITEM(list, count, value);
                        count++;

                    }

                    PyMem_Free(s);
                    break;
                }
                case Pickle::opcode::SHORT_BINUNICODE:{
                    PyObject *str;
                    Py_ssize_t size;
                    char *s;
                    if (UNLIKELY(frameLen_ - frameIdx_ < 1)) {
                        if((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
                            return -1;
                        *(&s) = shortBuf_;
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 1;
                    }
                    size = calc_binsize(s, 1);
                    if (size < 0) {
                        PyErr_Format(PyExc_OverflowError,
                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                     PY_SSIZE_T_MAX);
                        return -1;
                    }

                    bool newFlag = false;
                    if (UNLIKELY(frameLen_ - frameIdx_ < size)) {
                        s = (char *) PyMem_Malloc(size);
                        if (s == NULL)
                            return -1;
                        newFlag = true;
                        size_t begIdx=0, actualSize=0;
                        const size_t BUFFSIZE = 65536;
                        while(begIdx < size) {
                            actualSize = std::min(size - begIdx, BUFFSIZE);
                            if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                                return -1;
                            begIdx += actualSize;
                        }
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;
                    }
                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                    if (newFlag)
                        PyMem_Free(s);
                    if (str == NULL)
                        return -1;
                    PyList_SET_ITEM(list, count, str);
                    count++;
                    break;
                }
                case Pickle::opcode::BINUNICODE:{
                    PyObject *str;
                    Py_ssize_t size;
                    char *s;
                    if (UNLIKELY(frameLen_ - frameIdx_ < 4)) {
                        if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                            return -1;
                        *(&s) = shortBuf_;
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 4;
                    }
                    size = calc_binsize(s, 4);
                    if (size < 0) {
                        PyErr_Format(PyExc_OverflowError,
                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                     PY_SSIZE_T_MAX);
                        return -1;
                    }

                    bool newFlag = false;
                    if (UNLIKELY(frameLen_ - frameIdx_ < size)) {
                        s = (char *) PyMem_Malloc(size);
                        if (s == NULL)
                            return -1;
                        newFlag = true;
                        size_t begIdx=0, actualSize=0;
                        const size_t BUFFSIZE = 65536;
                        while(begIdx < size) {
                            actualSize = std::min(size - begIdx, BUFFSIZE);
                            if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                                return -1;
                            begIdx += actualSize;
                        }
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;
                    }
                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                    if (newFlag)
                        PyMem_Free(s);
                    if (str == NULL)
                        return -1;
                    PyList_SET_ITEM(list, count, str);
                    count++;
                    break;

                }
                case Pickle::opcode::BINUNICODE8:{
                    PyObject *str;
                    Py_ssize_t size;
                    char *s;
                    if (UNLIKELY(frameLen_ - frameIdx_ < 8)) {
                        if((ret = in_->readBytes(shortBuf_, 8, false)) != OK)
                            return -1;
                        *(&s) = shortBuf_;
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 8;
                    }
                    size = calc_binsize(s, 8);
                    if (size < 0) {
                        PyErr_Format(PyExc_OverflowError,
                                     "BINUNICODE exceeds system's maximum size of %zd bytes",
                                     PY_SSIZE_T_MAX);
                        return -1;
                    }

                    bool newFlag = false;
                    if (UNLIKELY(frameLen_ - frameIdx_ < size)) {
                        s = (char *) PyMem_Malloc(size);
                        if (s == NULL)
                            return -1;
                        newFlag = true;
                        size_t begIdx=0, actualSize=0;
                        const size_t BUFFSIZE = 65536;
                        while(begIdx < size) {
                            actualSize = std::min(size - begIdx, BUFFSIZE);
                            if((ret = in_->readBytes(s + begIdx, actualSize, actualSize)) != OK)
                                return -1;
                            begIdx += actualSize;
                        }
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += size;
                    }
                    str = PyUnicode_DecodeUTF8(s, size, "surrogatepass");
                    if (newFlag)
                        PyMem_Free(s);
                    if (str == NULL)
                        return -1;
                    PyList_SET_ITEM(list, count, str);
                    count++;
                    break;
                }
                case Pickle::opcode::FRAME: {
                    if(load_frame(ret) < 0)
                        return -1;
                    break;
                }
                case Pickle::opcode::APPENDS:
                    break;
                case Pickle::opcode::MEMOIZE: {
                    int size = unpickler_->memo_len + newObjCnt;
                    ++newObjCnt;
                    map.insert({size, count});
                    break;
                }
                case Pickle::opcode::BINGET:{
                    PyObject *value;
                    Py_ssize_t idx;
                    char *s;
                    if (UNLIKELY(frameLen_ - frameIdx_ < 1)) {
                        if((ret = in_->readBytes(shortBuf_, 1, false)) != OK)
                            return -1;
                        *(&s) = shortBuf_;
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 1;
                    }
                    idx = Py_CHARMASK(s[0]);
                    value = PyList_GetItem(list, map[idx] - 1);

                    if (value == NULL) {
                        PyObject *key = PyLong_FromSsize_t(idx);
                        if (key != NULL) {
                            PyErr_SetObject(PyExc_KeyError, key);
                            Py_DECREF(key);
                        }
                        return -1;
                    }
                    Py_IncRef(value);
                    PyList_SET_ITEM(list, count, value);
                    count++;
                    break;
                }
                case Pickle::opcode::LONG_BINGET:{
                    PyObject *value;
                    Py_ssize_t idx;
                    char *s;
                    if (UNLIKELY(frameLen_ - frameIdx_ < 4)) {
                        if((ret = in_->readBytes(shortBuf_, 4, false)) != OK)
                            return -1;
                        *(&s) = shortBuf_;
                    } else {
                        *(&s) = frame_ + frameIdx_;
                        frameIdx_ += 4;
                    }
                    idx = calc_binsize(s, 4);
                    value = PyList_GetItem(list, map[idx] - 1);

                    if (value == NULL) {
                        PyObject *key = PyLong_FromSsize_t(idx);
                        if (key != NULL) {
                            PyErr_SetObject(PyExc_KeyError, key);
                            Py_DECREF(key);
                        }
                        return -1;
                    }
                    Py_IncRef(value);
                    PyList_SET_ITEM(list, count, value);
                    count++;
                    break;
                }
                case Pickle::opcode::NEWTRUE:{
                    PyObject* value = Py_True;
                    Py_IncRef(value);
                    PyList_SET_ITEM(list, count, value);
                    count++;
                    break;
                }
                case Pickle::opcode::NEWFALSE:{
                    PyObject* value = Py_False;
                    Py_IncRef(value);
                    PyList_SET_ITEM(list, count, value);
                    count++;
                    break;
                }
                case Pickle::opcode::NONE:{
                    PyObject* value = Py_None;
                    Py_IncRef(value);
                    PyList_SET_ITEM(list, count, value);
                    count++;
                    break;
                }
                default: {
                    return -1;
                }
            }
            if (op == Pickle::opcode::APPENDS) {
                break;
            }
            if (op == Pickle::opcode::APPEND) {
                break;
            }
            if(frameIdx_ < frameLen_ ){
                op = frame_[frameIdx_];
                frameIdx_++;
            }else{
                if((ret = in_->readChar(op)) != OK)
                    return -1;
            }
        }
        Py_SIZE(unpickler_->stack) = x;
        return 0;
    }

    bool PickleUnmarshall::start(short flag, bool blocking, IO_ERR &ret) {
        if((ret = in_->readBytes(shortBuf_, 2, false)) != OK){
            return false;
        }
        unpickler_->num_marks = 0;
        unpickler_->stack->mark_set = 0;
        unpickler_->stack->fence = 0;
        if(shortBuf_[0] != Pickle::opcode::PROTO) {
            ret = INVALIDDATA;
            return false;
        }
        int i = (unsigned char)shortBuf_[1];
        if(i >  HIGHEST_PROTOCOL) {
            ret = INVALIDDATA;
            return false;
        }
        unpickler_->proto = i;
        if (Py_SIZE(unpickler_->stack))
            Pdata_clear(unpickler_->stack, 0);

        while (1) {
            char op;
            if(frameIdx_ < frameLen_ ){
                op = frame_[frameIdx_];
                frameIdx_++;
            }else{
                if((ret = in_->readChar(op)) != OK)
                    return false;
            }
            switch ((enum Pickle::opcode)op) {
                case Pickle::opcode::NONE: if(load_none() < 0) break; continue;
                case Pickle::opcode::BININT: if(load_binint(ret) < 0) break; continue;
                case Pickle::opcode::BININT1: if(load_binint1(ret) < 0) break; continue;
                case Pickle::opcode::BININT2: if(load_binint2(ret) < 0) break; continue;
                case Pickle::opcode::INT: if(load_int(ret) < 0) break; continue;
                case Pickle::opcode::LONG: if(load_long(ret) < 0) break; continue;
                case Pickle::opcode::LONG1: if(load_counted_long(1, ret) < 0) break; continue;
                case Pickle::opcode::LONG4: if(load_counted_long(4, ret) < 0) break; continue;
                case Pickle::opcode::FLOAT: if(load_float(ret) < 0) break; continue;
                case Pickle::opcode::BINFLOAT: if(load_binfloat(ret) < 0) break; continue;
                case Pickle::opcode::SHORT_BINBYTES: if(load_counted_binbytes(1, ret) < 0) break; continue;
                case Pickle::opcode::BINBYTES: if(load_counted_binbytes(4, ret) < 0) break; continue;
                case Pickle::opcode::BINBYTES8: if(load_counted_binbytes(8, ret) < 0) break; continue;
                case Pickle::opcode::SHORT_BINSTRING: if(load_counted_binstring(1, ret) < 0) break; continue;
                case Pickle::opcode::BINSTRING: if(load_counted_binstring(4, ret) < 0) break; continue;
                case Pickle::opcode::STRING: if(load_string(ret) < 0) break; continue;
                case Pickle::opcode::UNICODE: if(load_unicode(ret) < 0) break; continue;
                case Pickle::opcode::SHORT_BINUNICODE: if(load_counted_binunicode(1, ret) < 0) break; continue;
                case Pickle::opcode::BINUNICODE: if(load_counted_binunicode(4, ret) < 0) break; continue;
                case Pickle::opcode::BINUNICODE8: if(load_counted_binunicode(8, ret) < 0) break; continue;
                case Pickle::opcode::EMPTY_TUPLE: if(load_counted_tuple(0) < 0) break; continue;
                case Pickle::opcode::TUPLE1: if(load_counted_tuple(1) < 0) break; continue;
                case Pickle::opcode::TUPLE2: if(load_counted_tuple(2) < 0) break; continue;
                case Pickle::opcode::TUPLE3: if(load_counted_tuple(3) < 0) break; continue;
                case Pickle::opcode::TUPLE: if(load_tuple() < 0) break; continue;
                case Pickle::opcode::EMPTY_LIST: if(load_empty_list() < 0) break; continue;
                case Pickle::opcode::LIST: if(load_list() < 0) break; continue;
                case Pickle::opcode::EMPTY_DICT: if(load_empty_dict() < 0) break; continue;
                case Pickle::opcode::DICT: if(load_dict() < 0) break; continue;
                case Pickle::opcode::EMPTY_SET: if(load_empty_set() < 0) break; continue;
                case Pickle::opcode::ADDITEMS: if(load_additems() < 0) break; continue;
                case Pickle::opcode::FROZENSET: if(load_frozenset() < 0) break; continue;
                case Pickle::opcode::OBJ: if(load_obj() < 0) break; continue;
                case Pickle::opcode::INST: if(load_inst(ret) < 0) break; continue;
                case Pickle::opcode::NEWOBJ: if(load_newobj() < 0) break; continue;
                case Pickle::opcode::NEWOBJ_EX: if(load_newobj_ex() < 0) break; continue;
                case Pickle::opcode::GLOBAL: if(load_global(ret) < 0) break; continue;
                case Pickle::opcode::STACK_GLOBAL: if(load_stack_global() < 0) break; continue;
                case Pickle::opcode::APPEND: if(load_append() < 0) break; continue;
                case Pickle::opcode::APPENDS: if(load_appends() < 0) break; continue;
                case Pickle::opcode::BUILD: if(load_build() < 0) break; continue;
                case Pickle::opcode::DUP: if(load_dup() < 0) break; continue;
                case Pickle::opcode::BINGET: if(load_binget(ret) < 0) break; continue;
                case Pickle::opcode::LONG_BINGET: if(load_long_binget(ret) < 0) break; continue;
                case Pickle::opcode::GET: if(load_get(ret) < 0) break; continue;
                case Pickle::opcode::MARK: if(load_mark() < 0) break; continue;
                case Pickle::opcode::BINPUT: if(load_binput(ret) < 0) break; continue;
                case Pickle::opcode::LONG_BINPUT: if(load_long_binput(ret) < 0) break; continue;
                case Pickle::opcode::PUT: if(load_put(ret) < 0) break; continue;
                case Pickle::opcode::MEMOIZE: if(load_memoize() < 0) break; continue;
                case Pickle::opcode::POP: if(load_pop() < 0) break; continue;
                case Pickle::opcode::POP_MARK: if(load_pop_mark() < 0) break; continue;
                case Pickle::opcode::SETITEM: if(load_setitem() < 0) break; continue;
                case Pickle::opcode::SETITEMS: if(load_setitems() < 0) break; continue;
                case Pickle::opcode::PERSID: if(load_persid(ret) < 0) break; continue;
                case Pickle::opcode::BINPERSID: if(load_binpersid() < 0) break; continue;
                case Pickle::opcode::REDUCE: if(load_reduce() < 0) break; continue;
                case Pickle::opcode::PROTO: if(load_proto(ret) < 0) break; continue;
                case Pickle::opcode::FRAME: if(load_frame(ret) < 0) break; continue;
                case Pickle::opcode::EXT1: if(load_extension(1, ret) < 0) break; continue;
                case Pickle::opcode::EXT2: if(load_extension(2, ret) < 0) break; continue;
                case Pickle::opcode::EXT4: if(load_extension(4, ret) < 0) break; continue;
                case Pickle::opcode::NEWTRUE: if(load_bool(Py_True) < 0) break; continue;
                case Pickle::opcode::NEWFALSE: if(load_bool(Py_False) < 0) break; continue;
                case Pickle::opcode::SYMBOL: if(load_symbol(ret) < 0) return false; continue;
                case Pickle::opcode::OBJECTBEGIN: if(load_objectBegin(ret) < 0) return false;continue;

                case Pickle::opcode::STOP:
                    break;
                default:
                {
                    PickleState *st = _Pickle_GetGlobalState();
                    unsigned char c = (unsigned char) op;
                    if (0x20 <= c && c <= 0x7e && c != '\'' && c != '\\') {
                        PyErr_Format(st->UnpicklingError,
                                     "invalid load key, '%c'.", c);
                    }
                    else {
                        PyErr_Format(st->UnpicklingError,
                                     "invalid load key, '\\x%02x'.", c);
                    }
                    return false;
                }
            }
            break;                  /* and we are done! */
        }
        if(ret != OK) {
            return false;
        }
        if (PyErr_Occurred()) {
            return false;
        }
        if (_Unpickler_SkipConsumed(unpickler_) < 0)
            return false;
        PDATA_POP(unpickler_->stack, obj_);
        //Py_DecRef(obj_);
        return true;
    }

    void PickleUnmarshall::reset() {
        if(frame_) {
            delete [] frame_;
            frame_ = nullptr;
        }
        Unpickler_clear(unpickler_);
        Py_DECREF(unpickler_);
    }
};
