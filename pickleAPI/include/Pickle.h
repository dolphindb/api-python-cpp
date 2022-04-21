//
// Created by lin on 2020/11/6.
//

#ifndef __PICKLE_H
#define __PICKLE_H
#include "Python.h"
#include "DolphinDB.h"
#include "SysIO.h"

struct UnpicklerObject;
namespace dolphindb{
    class EXPORT_DECL PickleUnmarshall{
    public:
        PickleUnmarshall(const DataInputStreamSP& in);
        ~PickleUnmarshall(){}
        bool start(short flag, bool blocking, IO_ERR& ret);
        void reset();
        PyObject * getPyObj(){ return obj_; }
    private:
        int load_none();
        int load_int(IO_ERR& ret);
        int load_bool(PyObject *boolean);
        int load_binintx(char *s, size_t size);
        int load_binint(IO_ERR& ret);
        int load_binint1(IO_ERR& ret);
        int load_binint2(IO_ERR& ret);
        int load_long(IO_ERR& ret);
        int load_counted_long(size_t size, IO_ERR& ret);
        int load_float(IO_ERR& ret);
        int load_binfloat(IO_ERR& ret);
        int load_string(IO_ERR& ret);
        int load_counted_binstring(size_t nbytes, IO_ERR& ret);
        int load_counted_binbytes(size_t nbytes, IO_ERR& ret);
        int load_unicode(IO_ERR& ret);
        int load_counted_binunicode(size_t nbytes, IO_ERR& ret);
        int load_counted_tuple(Py_ssize_t len);
        int load_tuple();
        int load_empty_list();
        int load_empty_dict();
        int load_empty_set();
        int load_list();
        int load_dict();
        int load_frozenset();
        int load_obj();
        int load_inst(IO_ERR& ret);
        int load_newobj();
        int load_newobj_ex();
        int load_global(IO_ERR& ret);
        int load_stack_global();
        int load_persid(IO_ERR& ret);
        int load_binpersid();
        int load_pop();
        int load_pop_mark();
        int load_dup();
        int load_get(IO_ERR& ret);
        int load_binget(IO_ERR& ret);
        int load_long_binget(IO_ERR& ret);
        int load_extension(size_t nbytes, IO_ERR& ret);
        int load_put(IO_ERR& ret);
        int load_binput(IO_ERR& ret);
        int load_long_binput(IO_ERR& ret);
        int load_memoize();
        int load_append();
        int load_appends();
        int load_setitem();
        int load_setitems();
        int load_additems();
        int load_build();
        int load_mark();
        int load_reduce();
        int load_proto(IO_ERR& ret);
        int load_frame(IO_ERR& ret);
        int load_symbol(IO_ERR& ret, char &lastDoOp);
        int load_objectBegin(IO_ERR& ret);
    private:
        bool do_opr(char op, IO_ERR &ret);
        bool get_opr(char &op, IO_ERR &ret);
        PyObject * obj_;
        DataInputStreamSP in_;
        UnpicklerObject * unpickler_;
        char* frame_;
        char shortBuf_[8] = {0};
        size_t frameIdx_;
        size_t frameLen_;
    };
}; //end of namespace dolphindb

#endif //__PICKLE_H
