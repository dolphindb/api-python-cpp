#include "DdbPythonUtil.h"
#include "Concurrent.h"
#include "ConstantImp.h"
#include "ConstantMarshall.h"
#include "DolphinDB.h"
#include "ScalarImp.h"
#include "Util.h"
#include "Pickle.h"

namespace dolphindb {

const py::object Preserved::numpy_ = py::module::import("numpy");
//py::object Preserved::isnan_ = numpy_.attr("isnan");
//py::object Preserved::sum_ = numpy_.attr("sum");
const py::object Preserved::datetime64_ = numpy_.attr("datetime64");
const py::object Preserved::pandas_ = py::module::import("pandas");
//const std::string Preserved::pddataframe_=py::str(Preserved::pandas_.attr("DataFrame").get_type()).cast<std::string>();
const py::object Preserved::pddataframe_ = getType(pandas_.attr("DataFrame")());
const py::object Preserved::pdseries_ = getType(pandas_.attr("Series")());
const py::object Preserved::nparray_ = getType(py::array());
const py::object Preserved::npbool_ = getDType("bool");
const py::object Preserved::npint8_ = getDType("int8");
const py::object Preserved::npint16_ = getDType("int16");
const py::object Preserved::npint32_ = getDType("int32");
const py::object Preserved::npint64_ = getDType("int64");
const py::object Preserved::npfloat32_ = getDType("float32");
const py::object Preserved::npfloat64_ = getDType("float64");

py::object Preserved::npdatetime64M_(){ return getDType("datetime64[M]");}
py::object Preserved::npdatetime64D_(){ return getDType("datetime64[D]");}
py::object Preserved::npdatetime64m_(){ return getDType("datetime64[m]");}
py::object Preserved::npdatetime64s_(){ return getDType("datetime64[s]");}
py::object Preserved::npdatetime64h_(){ return getDType("datetime64[h]");}
py::object Preserved::npdatetime64ms_(){ return getDType("datetime64[ms]");}
py::object Preserved::npdatetime64us_(){ return getDType("datetime64[us]");}
py::object Preserved::npdatetime64ns_(){ return getDType("datetime64[ns]");}
py::object Preserved::npdatetime64_(){ return getDType("datetime64");}
const py::object Preserved::npobject_ = getDType("object");

const py::object Preserved::pynone_ = getType(py::none());
const py::object Preserved::pybool_ = getType(py::bool_());
const py::object Preserved::pyint_ = getType(py::int_());
const py::object Preserved::pyfloat_ = getType(py::float_());
const py::object Preserved::pystr_ = getType(py::str());
const py::object Preserved::pybytes_ = getType(py::bytes());
const py::object Preserved::pyset_ = getType(py::set());
const py::object Preserved::pytuple_ = getType(py::tuple());
const py::object Preserved::pylist_ = getType(py::list());
const py::object Preserved::pydict_ = getType(py::dict());

const static uint64_t npDoubleNan_ = 9221120237041090560LL;
static inline void SET_NPNAN(double *p, size_t len = 1) { std::fill((uint64_t *)p, ((uint64_t *)p) + len, npDoubleNan_); }
const static long long npLongNan_ = 0x8000000000000000;
static inline void SET_NPNAN(long long *p, size_t len = 1) { std::fill((long long *)p, ((long long *)p) + len, npLongNan_); }


#define DLOG //printf

py::object getPythonType(py::object &obj){
    if(py::hasattr(obj, "dtype")){
        py::object dtypeOfObj=py::getattr(obj, "dtype");
        //py::object dtname=py::getattr(dtypeOfObj, "name");
        //std::string name=py::str(dtname);
        //DLOG("getPythonType: %s.",name.data());
        return dtypeOfObj;
    }else{
        //DLOG("getPythonType failed.");
    }
    return Preserved::pynone_;
}

py::object DdbPythonUtil::toPython(ConstantSP obj, bool tableFlag) {
    if(obj.isNull()){
        DLOG("toPython NULL to None.");
        return py::none();
    }
    DLOG("{ toPython %s,%s %d flag %d.", Util::getDataTypeString(obj->getType()).data(),Util::getDataFormString(obj->getForm()).data(),obj->size(), tableFlag);
    if (obj.isNull() || obj->isNothing() || obj->isNull()) { return py::none(); }
    DATA_TYPE type = obj->getType();
    DATA_FORM form = obj->getForm();
    if (form == DF_VECTOR) {
        if(type == 128 + DT_SYMBOL){
            //SymbolVector
            FastSymbolVector *symbolVector=(FastSymbolVector*)obj.get();
            size_t size = symbolVector->size();
            py::array pyVec(py::dtype("object"), {size}, {});
            for (size_t i = 0; i < size; ++i) {
                py::str temp(symbolVector->getString(i));
                Py_IncRef(temp.ptr());
                memcpy(pyVec.mutable_data(i), &temp, sizeof(py::object));
            }
            return std::move(pyVec);
        }else if(type >= ARRAY_TYPE_BASE){
            //ArrayVector
            FastArrayVector *arrayVector=(FastArrayVector*)obj.get();
            size_t size = arrayVector->size();
            py::array pyVec(py::dtype("object"), {size}, {});
            for (size_t i = 0; i < size; ++i) {
                VectorSP subArray = arrayVector->get(i);
                py::object pySubVec = toPython(subArray, false);
                Py_IncRef(pySubVec.ptr());
                memcpy(pyVec.mutable_data(i), &pySubVec, sizeof(py::object));
            }
            return std::move(pyVec);
        }
        VectorSP ddbVec = obj;
        size_t size = ddbVec->size();
        DLOG("toPython vector %s size %d istable %d,",Util::getDataTypeString(type).data(),size,tableFlag);
        switch (type) {
            case DT_VOID: {
                py::array pyVec;
                pyVec.resize({size});
                return std::move(pyVec);
            }
            case DT_BOOL: {
                py::array pyVec(py::dtype("bool"), {size}, {});
                ddbVec->getBool(0, size, (char *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    // Play with the raw api of Python, be careful about the ref count
                    DLOG("has null.");
                    pyVec = pyVec.attr("astype")("object");
                    PyObject **p = (PyObject **)pyVec.mutable_data();
                    char buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getBool(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == INT8_MIN)) {
                                Py_DECREF(p[start + i]);
                                p[start + i] = Preserved::numpy_.attr("nan").ptr();
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_CHAR: {
                py::array pyVec(py::dtype("int8"), {size}, {});
                ddbVec->getChar(0, size, (char *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    pyVec = pyVec.attr("astype")("float64");
                    double *p = (double *)pyVec.mutable_data();
                    char buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getChar(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == INT8_MIN)) {
                                SET_NPNAN(p + start + i, 1);
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_SHORT: {
                py::array pyVec(py::dtype("int16"), {size}, {});
                ddbVec->getShort(0, size, (short *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    pyVec = pyVec.attr("astype")("float64");
                    double *p = (double *)pyVec.mutable_data();
                    short buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getShort(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == INT16_MIN)) {
                                SET_NPNAN(p + start + i, 1);
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_INT: {
                py::array pyVec(py::dtype("int32"), {size}, {});
                ddbVec->getInt(0, size, (int *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    pyVec = pyVec.attr("astype")("float64");
                    double *p = (double *)pyVec.mutable_data();
                    int buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getInt(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == INT32_MIN)) {
                                DLOG("set %d null.", i);
                                SET_NPNAN(p + start + i, 1);
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_LONG: {
                py::array pyVec(py::dtype("int64"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    pyVec = pyVec.attr("astype")("float64");
                    double *p = (double *)pyVec.mutable_data();
                    long long buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getLong(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == INT64_MIN)) {
                                SET_NPNAN(p + start + i, 1);
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_DATE: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 86400000000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 86400000000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[D]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_MONTH: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[M]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                        if(p[i] < 1970 * 12 || p[i] > 2262 * 12 + 3) {
                            throw RuntimeException("In dateFrame Month must between 1970.01M and 2262.04M");
                        }
                        p[i] -= 1970 * 12;
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[M]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                for (size_t i = 0; i < size; ++i) {
                    if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                    p[i] -= 1970 * 12;
                }
                return std::move(pyVec);
            }
            case DT_TIME: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 1000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 1000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[ms]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_MINUTE: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 60000000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 60000000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[m]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                //DLOG(".datetime64[m] array.");
                return std::move(pyVec);
            }
            case DT_SECOND: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 1000000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 1000000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[s]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_DATETIME: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 1000000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 1000000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[s]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_TIMESTAMP: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 1000000;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 1000000;
                        }
                    }
                    return std::move(pyVec);
                }
                py::array pyVec(py::dtype("datetime64[ms]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_NANOTIME: {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_NANOTIMESTAMP: {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                return std::move(pyVec);
            }
            case DT_DATEHOUR: {
                if(tableFlag) {
                    py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    long long *p = (long long *)pyVec.mutable_data();
                    if (UNLIKELY(ddbVec->hasNull())) {
                        DLOG("has null.");
                        for (size_t i = 0; i < size; ++i) {
                            if (UNLIKELY(p[i] == INT64_MIN)) { continue; }
                            p[i] *= 3600000000000ll;
                        }
                    }
                    else {
                        for (size_t i = 0; i < size; ++i) {
                            p[i] *= 3600000000000ll;
                        }
                    }
                    return std::move(pyVec);
                }
                else {
                    py::array pyVec(py::dtype("datetime64[h]"), {size}, {});
                    ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                    return std::move(pyVec);
                }
            }
            case DT_FLOAT: {
                py::array pyVec(py::dtype("float32"), {size}, {});
                ddbVec->getFloat(0, size, (float *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    auto p = (float *)pyVec.mutable_data();
                    float buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getFloat(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == FLT_NMIN)) {
                                p[i]=NAN;
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_DOUBLE: {
                py::array pyVec(py::dtype("float64"), {size}, {});
                ddbVec->getDouble(0, size, (double *)pyVec.mutable_data());
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    double *p = (double *)pyVec.mutable_data();
                    double buf[1024];
                    int start = 0;
                    int N = size;
                    while (start < N) {
                        int len = std::min(N - start, 1024);
                        ddbVec->getDouble(start, len, buf);
                        for (int i = 0; i < len; ++i) {
                            if(UNLIKELY(buf[i] == DBL_NMIN)) {
                                SET_NPNAN(p + start + i, 1);
                            }
                        }
                        start += len;
                    }
                }
                return std::move(pyVec);
            }
            case DT_IP:
            case DT_UUID:
            case DT_INT128:
            case DT_SYMBOL:
            case DT_STRING: {
                py::array pyVec(py::dtype("object"), {size}, {});
                for (size_t i = 0; i < size; ++i) {
                    py::str temp(ddbVec->getString(i));
                    Py_IncRef(temp.ptr());
                    memcpy(pyVec.mutable_data(i), &temp, sizeof(py::object));
                }
                return std::move(pyVec);
            }
            case DT_ANY: {
                // handle numpy.array of objects
                auto l = py::list();
                    for (size_t i = 0; i < size; ++i) { l.append(toPython(ddbVec->get(i))); }
                    return py::array(l);
            }
            default: {
                throw RuntimeException("type error in Vector convertion! ");
            };
        }
    } else if (form == DF_TABLE) {
        TableSP ddbTbl = obj;
        size_t columnSize = ddbTbl->columns();
        using namespace py::literals;
        py::array first = toPython(obj->getColumn(0), true);
        auto colName = py::list();
        colName.append(py::str(ddbTbl->getColumnName(0)));
        py::object dataframe = Preserved::pandas_.attr("DataFrame")(first, "columns"_a = colName);
        for (size_t i = 1; i < columnSize; ++i) {
            ConstantSP col = obj->getColumn(i);
            dataframe[ddbTbl->getColumnName(i).data()] = toPython(col, true);
        }
        return dataframe;
    } else if (form == DF_SCALAR) {
        switch (type) {
            case DT_VOID: return py::none();
            case DT_BOOL: return py::bool_(obj->getBool());
            case DT_CHAR:
            case DT_SHORT:
            case DT_INT:
            case DT_LONG: return py::int_(obj->getLong());
            case DT_DATE: return Preserved::datetime64_(obj->getLong(), "D");
            case DT_MONTH: return Preserved::datetime64_(obj->getLong() - 23640, "M");    // ddb starts from 0000.0M
            case DT_TIME: return Preserved::datetime64_(obj->getLong(), "ms");
            case DT_MINUTE: return Preserved::datetime64_(obj->getLong(), "m");
            case DT_SECOND: return Preserved::datetime64_(obj->getLong(), "s");
            case DT_DATETIME: return Preserved::datetime64_(obj->getLong(), "s");
            case DT_DATEHOUR: return Preserved::datetime64_(obj->getLong(), "h");
            case DT_TIMESTAMP: return Preserved::datetime64_(obj->getLong(), "ms");
            case DT_NANOTIME: return Preserved::datetime64_(obj->getLong(), "ns");
            case DT_NANOTIMESTAMP: return Preserved::datetime64_(obj->getLong(), "ns");
            case DT_FLOAT:
            case DT_DOUBLE: return py::float_(obj->getDouble());
            case DT_IP:
            case DT_UUID:
            case DT_INT128:
                //                    return py::bytes(reinterpret_cast<const char *>(obj->getBinary()));
            case DT_SYMBOL:
            case DT_BLOB:
            case DT_STRING: return py::str(obj->getString());
            default: throw RuntimeException("type error in Scalar convertion!");
        }
    } else if (form == DF_DICTIONARY) {
        DictionarySP ddbDict = obj;
        DATA_TYPE keyType = ddbDict->getKeyType();
        if (keyType != DT_STRING && keyType != DT_SYMBOL && ddbDict->keys()->getCategory() != INTEGRAL) {
            throw RuntimeException("currently only string, symbol or integral key is supported in dictionary");
        }
        VectorSP keys = ddbDict->keys();
        VectorSP values = ddbDict->values();
        py::dict pyDict;
        if (keyType == DT_STRING) {
            for (int i = 0; i < keys->size(); ++i) { pyDict[keys->getString(i).data()] = toPython(values->get(i)); }
        } else {
            for (int i = 0; i < keys->size(); ++i) { pyDict[py::int_(keys->getLong(i))] = toPython(values->get(i)); }
        }
        return pyDict;
    } else if (form == DF_MATRIX) {
        ConstantSP ddbMat = obj;
        size_t rows = ddbMat->rows();
        size_t cols = ddbMat->columns();
        // FIXME: currently only support numerical matrix
        if (ddbMat->getCategory() == MIXED) { throw RuntimeException("currently only support single typed matrix"); }
        ddbMat->setForm(DF_VECTOR);
        py::array pyMat = toPython(ddbMat);
        py::object pyMatRowLabel = toPython(ddbMat->getRowLabel());
        py::object pyMatColLabel = toPython(ddbMat->getColumnLabel());
        pyMat.resize({cols, rows});
        pyMat = pyMat.attr("transpose")();
        py::list pyMatList;
        pyMatList.append(pyMat);
        pyMatList.append(pyMatRowLabel);
        pyMatList.append(pyMatColLabel);
        return pyMatList;
    } else if (form == DF_PAIR) {
        VectorSP ddbPair = obj;
        py::list pyPair;
        for (int i = 0; i < ddbPair->size(); ++i) { pyPair.append(toPython(ddbPair->get(i))); }
        return pyPair;
    } else if (form == DF_SET) {
        VectorSP ddbSet = obj->keys();
        py::set pySet;
        for (int i = 0; i < ddbSet->size(); ++i) { pySet.add(toPython(ddbSet->get(i))); }
        return pySet;
    } else {
        throw RuntimeException("the form is not supported! ");
    }
}

DATA_TYPE numpyToDolphinDBType(py::array &array) {
    py::dtype type = array.dtype();
    //py::object target1=Preserved::npdatetime64ns_();
    //py::object target2=Preserved::npdatetime64h_();
    //DLOG("numpyToDolphinDBType type=%lld ns=%lld h=%lld. ",type.ptr(),target1.ptr(),target2.ptr());
    if (type.equal(Preserved::npbool_))
        return DT_BOOL;
    else if (type.equal(Preserved::npint8_))
        return DT_CHAR;
    else if (type.equal(Preserved::npint16_))
        return DT_SHORT;
    else if (type.equal(Preserved::npint32_))
        return DT_INT;
    else if (type.equal(Preserved::npint64_))
        return DT_LONG;
    else if (type.equal(Preserved::npfloat32_))
        return DT_FLOAT;
    else if (type.equal(Preserved::npfloat64_))
        return DT_DOUBLE;
    else if (type.equal(Preserved::npdatetime64D_()))
        return DT_DATE;
    else if (type.equal(Preserved::npdatetime64M_()))
        return DT_MONTH;
    else if (type.equal(Preserved::npdatetime64m_()))
        return DT_MINUTE;
    else if (type.equal(Preserved::npdatetime64s_()))
        return DT_DATETIME;
    else if (type.equal(Preserved::npdatetime64h_()))
        return DT_DATEHOUR;
    else if (type.equal(Preserved::npdatetime64ms_()))
        return DT_TIMESTAMP;
    else if (type.equal(Preserved::npdatetime64us_()))
        return DT_NANOTIMESTAMP;
    else if (type.equal(Preserved::npdatetime64ns_()))
        return DT_NANOTIMESTAMP;
    else if (type.equal(Preserved::npdatetime64_()))    // np.array of null datetime64
        return DT_DATETIME;
    else if (type.equal(Preserved::npobject_)){
        return DT_OBJECT;
    }else{
        return DT_OBJECT;
    }
}

static bool isValueNull(long long value){
    return value == npLongNan_;
}

static bool isValueNull(double value){
    uint64_t r;
    memcpy(&r, &value, sizeof(r));
    return r == npDoubleNan_;
}

ConstantSP DdbPythonUtil::toDolphinDBScalar(py::object obj, DATA_TYPE typeIndicator) {
    //DLOG("toDolphinDBScalar.start.");
    if(typeIndicator >= ARRAY_TYPE_BASE){//ArrayVector
        DATA_TYPE eleType = (DATA_TYPE)(typeIndicator - ARRAY_TYPE_BASE);
        ConstantSP dataVector;
        if(createVectorMatrix(obj, eleType, dataVector) == false){
            throw RuntimeException("Python data [" + py::str(obj.get_type()).cast<std::string>()+"] mismatch array vector type "+Util::getDataTypeString(eleType).data());
        }
        VectorSP anyVector = Util::createVector(DT_ANY, 0, 1);
	    anyVector->append(dataVector);
	    return anyVector;
    }
    if (py::isinstance(obj, Preserved::pynone_)) {
        return Util::createNullConstant(typeIndicator);
    } else if (py::isinstance(obj, Preserved::pybool_)) {
        auto result = obj.cast<bool>();
        return Util::createObject(typeIndicator, result);
    } else if (py::isinstance(obj, Preserved::pyint_)) {
        auto result = obj.cast<long long>();
        if (isValueNull(result)) {
            //DLOG("toDolphinDBScalar None int.");
            return Util::createNullConstant(typeIndicator);
        }
        return Util::createObject(typeIndicator, result);
    } else if (py::isinstance(obj, Preserved::pyfloat_)) {
        auto result = obj.cast<double>();
        if (isValueNull(result)) {
            //DLOG("toDolphinDBScalar None float.");
            return Util::createNullConstant(typeIndicator);
        }
        return Util::createObject(typeIndicator, result);
    } else if (py::isinstance(obj, Preserved::pystr_)) {
        auto result = obj.cast<std::string>();
        return Util::createObject(typeIndicator, result);
    } else if (py::isinstance(obj, Preserved::pybytes_)) {
        auto result = obj.cast<std::string>();
        return Util::createObject(typeIndicator, result);
    } else if (py::isinstance(obj, Preserved::datetime64_)) {
        //DLOG("toDolphinDBScalar.datetime64.start.");
        py::object type = getPythonType(obj);
        long long value=obj.attr("astype")("int64").cast<long long>();
        if(isValueNull(value)){
            DLOG("None toDolphinDBScalar.datetime64.");
            return Util::createNullConstant(typeIndicator);
        }
        //DLOG("toDolphinDBScalar.datetime64.end.");
        ConstantSP constobj;
        if (type.equal(Preserved::npdatetime64ns_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64ns_. ");
            constobj=Util::createNanoTimestamp(value);
        } else if (type.equal(Preserved::npdatetime64D_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64D_. ");
            constobj=Util::createDate(value);
        }else if (type.equal(Preserved::npdatetime64M_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64M_. ");
            constobj=Util::createMonth(1970, 1 + value);
        } else if (type.equal(Preserved::npdatetime64m_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64m_. ");
            constobj=Util::createMinute(value);
        } else if (type.equal(Preserved::npdatetime64s_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64s_. ");
            constobj=Util::createDateTime(value);
        } else if (type.equal(Preserved::npdatetime64h_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64h_. ");
            constobj=Util::createDateHour(value);
        } else if (type.equal(Preserved::npdatetime64ms_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64ms_. ");
            constobj=Util::createTimestamp(value);
        } else if (type.equal(Preserved::npdatetime64us_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64us_. ");
            constobj=Util::createNanoTimestamp(value * 1000ll);
        }else if (type.equal(Preserved::npdatetime64_())) {
            //DLOG("toDolphinDBScalar_Pythontype npdatetime64_. ");
            constobj = Util::createObject(typeIndicator, value);
        } else {
            throw RuntimeException("unsupported scalar numpy.datetime64 ["+ py::str(obj.get_type()).cast<std::string>()+"].");
        }
        //DLOG("toDolphinDBScalar.castTemporal.%s.start.",constobj->getString().data());
        constobj = constobj->castTemporal(typeIndicator);
        //DLOG("toDolphinDBScalar.castTemporal.%s.end.",constobj->getString().data());
        return constobj;
    }
    //DLOG("toDolphinDBScalar.unknow.start.");
    py::object type = getPythonType(obj);
    //DLOG("toDolphinDBScalar.unknow.end.");
    if (type.equal(Preserved::npbool_)){
        auto result = obj.cast<bool>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npint8_)) {
        auto result = obj.cast<short>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npint16_)) {
        auto result = obj.cast<int>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npint32_)) {
        auto result = obj.cast<int>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npint64_)) {
        auto result = obj.cast<long long>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npfloat32_)) {
        auto result = obj.cast<float>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npfloat64_)) {
        auto result = obj.cast<double>();
        return Util::createObject(typeIndicator, result);
    } else if (type.equal(Preserved::npdatetime64_())) {
        auto result = obj.cast<long long>();
        return Util::createObject(typeIndicator, result);
    } else {
        throw RuntimeException("unrecognized scalar Python data [" + py::str(obj.get_type()).cast<std::string>()+"].");
    }
}

DATA_TYPE toDolphinDBDataType(py::object obj, bool &isnull) {
    isnull=false;
    if (py::isinstance(obj, Preserved::pynone_)) {
        isnull=true;
        return DATA_TYPE::DT_DOUBLE;
    } else if (py::isinstance(obj, Preserved::pybool_)) {
        return DATA_TYPE::DT_BOOL;
    } else if (py::isinstance(obj, Preserved::pyint_)) {
        auto result = obj.cast<long long>();
        isnull=isValueNull(result);
        return DATA_TYPE::DT_LONG;
    } else if (py::isinstance(obj, Preserved::pyfloat_)) {
        auto result = obj.cast<double>();
        isnull=isValueNull(result);
        return DATA_TYPE::DT_DOUBLE;
    } else if (py::isinstance(obj, Preserved::pystr_)) {
        return DATA_TYPE::DT_STRING;
    } else if (py::isinstance(obj, Preserved::pybytes_)) {
        return DATA_TYPE::DT_BLOB;
    } else if (py::isinstance(obj, Preserved::datetime64_)) {
        //DLOG("toDolphinDBDataType.datetime64.start.");
        py::object type = getPythonType(obj);
        long long value=obj.attr("astype")("int64").cast<long long>();
        isnull=isValueNull(value);
        //DLOG("toDolphinDBDataType.datetime64.end.");
        //py::object target1=Preserved::npdatetime64ns_;
        //py::object target2=Preserved::npdatetime64h_;
        //DLOG("toDolphinDBDataType type=%lld ns=%lld h=%lld. ",type.ptr(),target1.ptr(),target2.ptr());
        if (type.equal(Preserved::npdatetime64ns_())) {
            //DLOG("toDolphinDBDataType npdatetime64ns_. ");
            return DATA_TYPE::DT_NANOTIMESTAMP;
        } else if (type.equal(Preserved::npdatetime64D_())) {
            //DLOG("toDolphinDBDataType npdatetime64D_. ");
            return DATA_TYPE::DT_DATE;
        } else if (type.equal(Preserved::npdatetime64M_())) {
            //DLOG("toDolphinDBDataType npdatetime64M_. ");
            return DATA_TYPE::DT_MONTH;
        } else if (type.equal(Preserved::npdatetime64m_())) {
            return DATA_TYPE::DT_MINUTE;
        } else if (type.equal(Preserved::npdatetime64s_())) {
            return DATA_TYPE::DT_DATETIME;
        } else if (type.equal(Preserved::npdatetime64h_())) {
            return DATA_TYPE::DT_DATEHOUR;
        } else if (type.equal(Preserved::npdatetime64ms_())) {
            return DATA_TYPE::DT_TIMESTAMP;
        } else if (type.equal(Preserved::npdatetime64us_())) {
            return DATA_TYPE::DT_NANOTIMESTAMP;
        }else if (type.equal(Preserved::npdatetime64_())) {
            return DATA_TYPE::DT_DATETIME;
        } else {
            //DLOG("unsupported python data type numpy.datetime64 [%s].",py::str(obj.get_type()).cast<std::string>().data());
            return DATA_TYPE::DT_OBJECT;
        }
    }
    //DLOG("toDolphinDBDataType.unknow.start.");
    py::object type = getPythonType(obj);
    //DLOG("toDolphinDBDataType.unknow.end.");
    if (type.equal(Preserved::npbool_)) {
        return DATA_TYPE::DT_BOOL;
    } else if (type.equal(Preserved::npint8_)) {
        return DATA_TYPE::DT_SHORT;
    } else if (type.equal(Preserved::npint16_)) {
        return DATA_TYPE::DT_INT;
    } else if (type.equal(Preserved::npint32_)) {
        return DATA_TYPE::DT_INT;
    } else if (type.equal(Preserved::npint64_)) {
        return DATA_TYPE::DT_LONG;
    } else if (type.equal(Preserved::npfloat32_)) {
        return DATA_TYPE::DT_FLOAT;
    } else if (type.equal(Preserved::npfloat64_)) {
        return DATA_TYPE::DT_DOUBLE;
    } else if (type.equal(Preserved::npdatetime64_())) {
        return DATA_TYPE::DT_DATETIME;
    } else {
        //DLOG("unsupported python data type [%s].",py::str(obj.get_type()).cast<std::string>().data());
        return DATA_TYPE::DT_OBJECT;
    }
}

bool isObjArray(py::object &obj){
    return py::isinstance(obj, Preserved::pytuple_)||
            py::isinstance(obj, Preserved::pylist_)||
            py::isinstance(obj, Preserved::nparray_)||
            py::isinstance(obj, Preserved::pdseries_);
}

bool DdbPythonUtil::createVectorMatrix(py::object obj, DATA_TYPE typeIndicator, ConstantSP &ddbvector){
    vector<py::object> children;
    size_t rows, cols;
    DATA_TYPE type = typeIndicator;
    if(py::isinstance(obj, Preserved::pytuple_)){
        DLOG("pytuple_ start.");
        py::tuple pyVec = obj;
        rows = 1;
        cols = pyVec.size();
        children.resize(pyVec.size());
        int index=0;
        for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {//check if child is array
            children[index++]=py::reinterpret_borrow<py::object>(*it);
        }
        //DLOG("pytuple_ end.");
    }else if(py::isinstance(obj, Preserved::pylist_)){
        DLOG("pylist_ start.");
        py::list pyVec = obj;
        rows = 1;
        cols = pyVec.size();
        children.resize(pyVec.size());
        int index=0;
        for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {//check if child is array
            children[index++]=py::reinterpret_borrow<py::object>(*it);
        }
        //DLOG("pylist_ end.");
    }else if(py::isinstance(obj, Preserved::nparray_)
            ||py::isinstance(obj, Preserved::pdseries_)){
        //std::string text=py::str(obj.get_type());
        //DLOG("nparrayseries_ %s start.", text.data());
        py::array pyVec = obj;
        int dim = pyVec.ndim();
        DLOG("nparrayseries_dim %d.", dim);
        if(dim == 1){
            rows = 1;
            cols = pyVec.size();
        }else if(dim == 2){
            rows = pyVec.shape(0);
            cols = pyVec.shape(1);
            pyVec = Preserved::numpy_.attr("array")(pyVec);
            pyVec = pyVec.attr("transpose")().attr("reshape")(pyVec.size());
        }else{
            throw RuntimeException("numpy.ndarray with dimension > 2 is not supported");
        }
        children.resize(pyVec.size());
        int index=0;
        for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {//check if child is array
            children[index++]=py::reinterpret_borrow<py::object>(*it);
        }
        if(type == DT_OBJECT){
            type = numpyToDolphinDBType(pyVec);
            //DLOG("VectorNumpy_numpyToDolphinDBType: %s.",Util::getDataTypeString(type).data());
        }
        //DLOG("nparrayseries_ end.");
    }else{
        return false;
    }
    if(type == DT_OBJECT){//unknow type
        bool isnull;
        DATA_TYPE curType, nullType = DT_OBJECT;
        for(auto &one : children){
            if(isObjArray(one)){
                type = DATA_TYPE::DT_ANY;
                break;
            }
            if(py::isinstance(one, Preserved::pynone_))//null can be any type, ignore it
                continue;
            curType = toDolphinDBDataType(one,isnull);
            DLOG("Vector_toDolphinDBDataType: %s is null %d.",Util::getDataTypeString(curType).data(),isnull);
            if(isnull){
                nullType = curType;
                //keep type last value.
                continue;
            }
            if(curType == DT_OBJECT){
                DLOG("toAny for Object type.");
                type = DATA_TYPE::DT_ANY;
                break;
            }
            if(type == DT_OBJECT){//type is default value, set it
                type = curType;
                continue;
            }
            if(type != curType){//two types, set any
                DLOG("toAny for %s!=%s.",Util::getDataTypeString(type).data(),Util::getDataTypeString(curType).data());
                type = DATA_TYPE::DT_ANY;
                break;
            }
        }
        if(type == DT_OBJECT){//Is all none???
            if(nullType != DT_OBJECT){
                type = nullType;
                DLOG("all null, set to last null type %s.",Util::getDataTypeString(nullType).data());
            }else{
                DLOG("toAny for All Object type.");
                type = DT_DOUBLE;
            }
        }
    }
    DLOG("createVectorMatrix type %s shape %d*%d size %d start.",Util::getDataTypeString(type).data(),rows,cols,children.size());
    if(rows <= 1){//vector
        //DLOG("createVector_%s.",Util::getDataTypeString(type).data());
        VectorSP ddbVec = Util::createVector(type, 0, children.size());
        DATA_TYPE objType = type;
        if(objType == DT_ANY)
            objType = DT_OBJECT;
        for (auto &pyobj : children) {
            //DLOG("toDolphinDB.Vector3_1.");
            ConstantSP item = toDolphinDB(pyobj, DF_CHUNK, objType);
            //DLOG("toDolphinDB.Vector3_2.");
            ddbVec->append(item);
        }
        //DLOG("vector %s %d.",Util::getDataTypeString(type).data(),(int)children.size());
        ddbvector=ddbVec;
    }else{//matrix
        //DLOG("createMatrix_%s.",Util::getDataTypeString(type).data());
        ConstantSP ddbMat = Util::createMatrix(type, cols, rows, cols);
        //DLOG("toDolphinDB.Matrix2.");
        DATA_TYPE objType = type;
        if(objType == DT_ANY)
            objType = DT_OBJECT;
        int index = 0;
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                //DLOG("toDolphinDB.Matrix3.");
                ddbMat->set(i, j, toDolphinDB(children[index++], DF_CHUNK, objType));
                //DLOG("toDolphinDB.Matrix4.");
            }
        }
        //DLOG("matrix %s %d,%d.",Util::getDataTypeString(type).data(),(int)rows,(int)cols);
        //DLOG("toDolphinDB.Matrix9.");
        ddbvector=ddbMat;
    }
    DLOG("createVectorMatrix type %s shape %d*%d size %d end.",Util::getDataTypeString(type).data(),rows,cols,children.size());
    return true;
}

ConstantSP DdbPythonUtil::toDolphinDB(py::object obj, DATA_FORM formIndicator, DATA_TYPE typeIndicator) {
    DLOG("{ toDolphinDB:");
     //if(formIndicator!=DF_CHUNK||typeIndicator!=DT_OBJECT){
     //    DLOG("toDolphinDB start %s,%s: ",Util::getDataTypeString(typeIndicator).data(),
     //                                Util::getDataFormString(formIndicator).data());
     //}
    ConstantSP ddbConst;
    if(createVectorMatrix(obj, typeIndicator, ddbConst) == false){//it's not vector
        if(formIndicator == DF_VECTOR){// Exception: vector is expected
            throw RuntimeException("Unexpected DF_VECTOR form type "+py::str(obj.get_type()).cast<std::string>());
        }
        //std::string type=py::str(obj.get_type()).cast<std::string>();
        //std::string pdtypestr=py::str(Preserved::pddataframe_).cast<std::string>();
        //DLOG("toDolphinDB pddataframe_ %s,%s.",type.data(),pdtypestr.data());
        if (py::isinstance(obj, Preserved::pddataframe_)) {
        //if(pdtypestr==type){
            DLOG("{ toDolphinDB.pddataframe.start ");
            py::object dataframe = obj;
            py::object pyLabel = dataframe.attr("columns");
            py::dict typeIndicators = py::getattr(dataframe, "__DolphinDB_Type__", py::dict());
            size_t columnSize = pyLabel.attr("size").cast<size_t>();
            vector<std::string> columnNames;
            columnNames.reserve(columnSize);

            //static py::object stringType = py::eval("str");

            //DLOG("toDolphinDB.pddataframe.2");
            for (auto it = pyLabel.begin(); it != pyLabel.end(); ++it) {
                if (!py::isinstance(*it, Preserved::pystr_)) {
                    throw RuntimeException("DolphinDB only support string as column names, and each of them must be a valid variable name.");
                }
                auto cur = it->cast<string>();
                // if (!Util::isVariableCandidate(cur)) {
                //     throw RuntimeException("'" + cur + "' is not a valid variable name, thus can not be used as a column name in DolphinDB.");
                // }
                columnNames.emplace_back(cur);
            }
            //DLOG("toDolphinDB.pddataframe.3");

            vector<ConstantSP> columns;
            columns.reserve(columnSize);
            for (size_t i = 0; i < columnSize; ++i) {
                DLOG("pddataframe column %d." , i);
                if (typeIndicators.contains(columnNames[i].data())) {
                    DATA_TYPE type = static_cast<DATA_TYPE>(typeIndicators[columnNames[i].data()].cast<int>());
                    columns.emplace_back(toDolphinDB(py::array(dataframe[columnNames[i].data()]), DF_VECTOR, type));
                } else {
                    columns.emplace_back(toDolphinDB(py::array(dataframe[columnNames[i].data()])));
                }
            }
            TableSP ddbTbl = Util::createTable(columnNames, columns);
            DLOG("toDolphinDB.pddataframe.end. }");
            //DLOG("toDolphinDB.pddataframe.4");
            ddbConst = ddbTbl;
        } else if (py::isinstance(obj, Preserved::pyset_)) {
            DLOG("{ toDolphinDB.pyset start.");
            py::set pySet = obj;
            vector<ConstantSP> _ddbSet;
            DATA_TYPE type = DT_VOID;
            DATA_FORM form = DF_SCALAR;
            int types = 0;
            int forms = 1;
            for (auto it = pySet.begin(); it != pySet.end(); ++it) {
                _ddbSet.push_back(toDolphinDB(py::reinterpret_borrow<py::object>(*it)));
                if (_ddbSet.back()->isNull()) { continue; }
                DATA_TYPE tmpType = _ddbSet.back()->getType();
                DATA_FORM tmpForm = _ddbSet.back()->getForm();
                if (tmpType != type) {
                    types++;
                    type = tmpType;
                }
                if (tmpForm != form) { forms++; }
            }
            if (types >= 2 || forms >= 2) {
                throw RuntimeException("set in DolphinDB doesn't support multiple types");
            } else if (types == 0) {
                throw RuntimeException("can not create all None set");
            }
            SetSP ddbSet = Util::createSet(type, _ddbSet.size());
            for (auto &v : _ddbSet) { ddbSet->append(v); }
            //DLOG("toDolphinDB.pyset.9");
            DLOG(" toDolphinDB.pyset end.}");
            ddbConst = ddbSet;
        } else if (py::isinstance(obj, Preserved::pydict_)) {
            DLOG("{ toDolphinDB.pydict start.");
            py::dict pyDict = obj;
            size_t size = pyDict.size();
            vector<ConstantSP> _ddbKeyVec;
            vector<ConstantSP> _ddbValVec;
            DATA_TYPE keyType = DT_VOID;
            DATA_TYPE valType = DT_VOID;
            DATA_FORM keyForm = DF_SCALAR;
            DATA_FORM valForm = DF_SCALAR;
            int keyTypes = 0;
            int valTypes = 0;
            int keyForms = 1;
            int valForms = 1;
            for (auto it = pyDict.begin(); it != pyDict.end(); ++it) {
                _ddbKeyVec.push_back(toDolphinDB(py::reinterpret_borrow<py::object>(it->first)));
                _ddbValVec.push_back(toDolphinDB(py::reinterpret_borrow<py::object>(it->second)));
                if (_ddbKeyVec.back()->isNull() || _ddbValVec.back()->isNull()) { continue; }
                DATA_TYPE tmpKeyType = _ddbKeyVec.back()->getType();
                DATA_TYPE tmpValType = _ddbValVec.back()->getType();
                DATA_FORM tmpKeyForm = _ddbKeyVec.back()->getForm();
                DATA_FORM tmpValForm = _ddbValVec.back()->getForm();
                if (tmpKeyType != keyType) {
                    keyTypes++;
                    keyType = tmpKeyType;
                }
                if (tmpValType != valType) {
                    valTypes++;
                    valType = tmpValType;
                }
                if (tmpKeyForm != keyForm) { keyForms++; }
                if (tmpValForm != valForm) { valForms++; }
            }
            if (keyTypes >= 2 || keyType == DT_BOOL || keyForms >= 2) { throw RuntimeException("the key type can not be BOOL or ANY"); }
            if (valTypes >= 2 || valForms >= 2) {
                valType = DT_ANY;
            } else if (keyTypes == 0 || valTypes == 0) {
                throw RuntimeException("can not create all None vector in dictionary");
            }
            VectorSP ddbKeyVec = Util::createVector(keyType, 0, size);
            VectorSP ddbValVec = Util::createVector(valType, 0, size);
            for (size_t i = 0; i < size; ++i) {
                ddbKeyVec->append(_ddbKeyVec[i]);
                ddbValVec->append(_ddbValVec[i]);
            }
            DictionarySP ddbDict = Util::createDictionary(keyType, valType);
            ddbDict->set(ddbKeyVec, ddbValVec);
            //DLOG("toDolphinDB.pydict.9");
            DLOG(" toDolphinDB.pydict end.}");
            ddbConst = ddbDict;
        }else{//Scalar
            if(typeIndicator == DT_OBJECT){
                //DLOG("toDolphindb_toDolphinDBDataType start.");
                bool isnull;
                typeIndicator = toDolphinDBDataType(obj,isnull);
                //DLOG("toDolphindb_toDolphinDBDataType: %s.",Util::getDataTypeString(typeIndicator).data());
                if(typeIndicator == DT_OBJECT){
                    throw RuntimeException("DolphinDB doesn't support Python data " + py::str(obj.get_type()).cast<std::string>());
                }
            }
            //DLOG("toDolphinDB.toDolphinDBScalar.start.");
            DLOG("toDolphindb_toDolphinDBScalar: %s.",Util::getDataTypeString(typeIndicator).data());
            ddbConst=toDolphinDBScalar(obj, typeIndicator);
        }
    }
    //DLOG("toDolphinDB.end.\n");
    DLOG(" toDolphinDB %s,%s %d.}", Util::getDataTypeString(ddbConst->getType()).data(),Util::getDataFormString(ddbConst->getForm()).data(),ddbConst->size()); if(ddbConst->size()>1) DLOG("\n");
    return ddbConst;
}

}
