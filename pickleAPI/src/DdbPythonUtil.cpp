#include "DdbPythonUtil.h"
#include "Concurrent.h"
#include "ConstantImp.h"
#include "ConstantMarshall.h"
#include "DolphinDB.h"
#include "ScalarImp.h"
#include "Util.h"
#include "Pickle.h"
#include "MultithreadedTableWriter.h"

namespace dolphindb {

const py::object Preserved::numpy_ = py::module::import("numpy");
//py::object Preserved::isnan_ = numpy_.attr("isnan");
//py::object Preserved::sum_ = numpy_.attr("sum");
const py::object Preserved::datetime64_ = numpy_.attr("datetime64");
const py::object Preserved::pandas_ = py::module::import("pandas");
//const std::string Preserved::pddataframe_=py::str(Preserved::pandas_.attr("DataFrame").get_type()).cast<std::string>();
const py::object Preserved::pddataframe_ = getType(pandas_.attr("DataFrame")());
const py::object Preserved::pdNaT = getType(pandas_.attr("NaT"));
const py::object Preserved::pdseries_ = getType(pandas_.attr("Series")(py::dtype("float64")));
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


#define DLOG //DLogger::Info
#define RECORDTIME //RecordTime _recordTime

py::object getPythonType(const py::object &obj){
    if(py::hasattr(obj, "dtype")){
        py::object dtypeOfObj=py::getattr(obj, "dtype");
        //py::object dtname=py::getattr(dtypeOfObj, "name");
        //std::string name=py::str(dtname);
        //DLOG("getPythonType: %s.",name.data());
        return dtypeOfObj;
    }
    return Preserved::pynone_;
}

void DdbPythonUtil::createPyVector(const ConstantSP &obj,py::object &pyObject,bool tableFlag,const ToPythonOption *poption){
    //RECORDTIME("createPyVector");
    VectorSP ddbVec = obj;
    size_t size = ddbVec->size();
    DATA_TYPE type = obj->getType();
    //DLOG("toPython vector",Util::getDataTypeString(type).data(),size,tableFlag);
    switch (type) {
        case DT_VOID: {
            py::array pyVec(py::dtype("object"));
            pyVec.resize({size});
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
        }
        case DT_DATE: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)){
                            SET_NPNAN(p + i,1);
                        }else{
                            p[i] *= 86400000000000;
                        }
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 86400000000000;
                    }
                }
                pyObject=std::move(pyVec);
            }else{
                py::array pyVec(py::dtype("datetime64[D]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_MONTH: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[M]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                for (size_t i = 0; i < size; ++i) {
                    if (UNLIKELY(p[i] == INT64_MIN)) {
                        SET_NPNAN(p + i,1);
                        continue;
                    }
                    if(p[i] < 1970 * 12 || p[i] > 2262 * 12 + 3) {
                        throw RuntimeException("In dateFrame Month must between 1970.01M and 2262.04M");
                    }
                    p[i] -= 1970 * 12;
                }
                pyObject=std::move(pyVec);
            }else{
                py::array pyVec(py::dtype("datetime64[M]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                for (size_t i = 0; i < size; ++i) {
                    if (UNLIKELY(p[i] == INT64_MIN)) {
                        SET_NPNAN(p + i,1);
                        continue;
                    }
                    p[i] -= 1970 * 12;
                }
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_TIME: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 1000000;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 1000000;
                    }
                }
                pyObject=std::move(pyVec);
            }
            else{
                py::array pyVec(py::dtype("datetime64[ms]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_MINUTE: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 60000000000;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 60000000000;
                    }
                }
                pyObject=std::move(pyVec);
            }else{
                py::array pyVec(py::dtype("datetime64[m]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                //DLOG(".datetime64[m] array.");
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_SECOND: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 1000000000;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 1000000000;
                    }
                }
                pyObject=std::move(pyVec);
            }
            else{
                py::array pyVec(py::dtype("datetime64[s]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_DATETIME: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 1000000000;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 1000000000;
                    }
                }
                pyObject=std::move(pyVec);
            }else{
                py::array pyVec(py::dtype("datetime64[s]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_TIMESTAMP: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 1000000;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 1000000;
                    }
                }
                pyObject=std::move(pyVec);
            }else{
                py::array pyVec(py::dtype("datetime64[ms]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
        }
        case DT_NANOTIME: {
            py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
            ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
            pyObject=std::move(pyVec);
            break;
        }
        case DT_NANOTIMESTAMP: {
            py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
            ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
            pyObject=std::move(pyVec);
            break;
        }
        case DT_DATEHOUR: {
            if(tableFlag) {
                py::array pyVec(py::dtype("datetime64[ns]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                long long *p = (long long *)pyVec.mutable_data();
                if (UNLIKELY(ddbVec->hasNull())) {
                    DLOG("has null.");
                    for (size_t i = 0; i < size; ++i) {
                        if (UNLIKELY(p[i] == INT64_MIN)) {
                            SET_NPNAN(p + i,1);
                            continue;
                        }
                        p[i] *= 3600000000000ll;
                    }
                }
                else {
                    for (size_t i = 0; i < size; ++i) {
                        p[i] *= 3600000000000ll;
                    }
                }
                pyObject=std::move(pyVec);
            }
            else {
                py::array pyVec(py::dtype("datetime64[h]"), {size}, {});
                ddbVec->getLong(0, size, (long long *)pyVec.mutable_data());
                pyObject=std::move(pyVec);
            }
            break;
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
                            p[i] = NAN;
                        }
                    }
                    start += len;
                }
            }
            pyObject=std::move(pyVec);
            break;
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
            pyObject=std::move(pyVec);
            break;
        }
        case DT_IP:
        case DT_UUID:
        case DT_INT128:
        case DT_SYMBOL:
        case DT_STRING: {
            py::array pyVec(py::dtype("object"), {size}, {});
            for (size_t i = 0; i < size; ++i) {
                // string &str=ddbVec->getString(i);
                // PyObject *pstr = PyBytes_FromStringAndSize(str.data(),str.size());
                // memcpy(pyVec.mutable_data(i), pstr, sizeof(PyObject*));

                py::str temp(ddbVec->getString(i));
                Py_IncRef(temp.ptr());
                memcpy(pyVec.mutable_data(i), &temp, sizeof(py::object));
            }
            pyObject=std::move(pyVec);
            break;
        }
        case DT_ANY: {
            // handle numpy.array of objects
            py::list list(size);
            for (size_t i = 0; i < size; ++i) {
                list[i]=toPython(ddbVec->get(i),false,poption);
            }
            pyObject = std::move(list);
            break;
        }
        default: {
            throw RuntimeException("type error in Vector convertion! ");
        };
    }
}

py::object DdbPythonUtil::toPython(ConstantSP obj,bool tableFlag,const ToPythonOption *poption) {
    //RECORDTIME("toPython");
    if (obj.isNull() || obj->isNothing() || obj->isNull()){
        DLOG("{ toPython NULL to None. }");
        return py::none();
    }
    DLOG("{ toPython", Util::getDataTypeString(obj->getType()).data(),Util::getDataFormString(obj->getForm()).data(),obj->size(),"table",tableFlag);
    static const ToPythonOption defaultOption;
    if(poption==NULL)
        poption=&defaultOption;
    DATA_TYPE type = obj->getType();
    DATA_FORM form = obj->getForm();
    py::object pyObject;
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
            pyObject = std::move(pyVec);
        }else if(type >= ARRAY_TYPE_BASE){
            //ArrayVector
            FastArrayVector *arrayVector=(FastArrayVector*)obj.get();
            if(poption->table2List==false){
                DLOG("arrayvector to df",type);
                size_t size = arrayVector->size();
                py::array pyVec(py::dtype("object"), {size}, {});
                for (size_t i = 0; i < size; ++i) {
                    VectorSP subArray = arrayVector->get(i);
                    py::object pySubVec = toPython(subArray, tableFlag, poption);
                    Py_IncRef(pySubVec.ptr());
                    memcpy(pyVec.mutable_data(i), &pySubVec, sizeof(py::object));
                }
                pyObject = std::move(pyVec);
            }else{
                DLOG("arrayvector to list",type);
                size_t cols = arrayVector->checkVectorSize();
                if(cols < 0){
                    throw RuntimeException("array vector can't convert to a 2D array!");
                }
                VectorSP valueSP = arrayVector->getFlatValueArray();
                py::object py1darray;
                createPyVector(valueSP, py1darray, tableFlag, poption);
                py::array py2DVec(py1darray);
                size_t rows = arrayVector->rows();
                py2DVec.resize( {rows, cols} );
                pyObject = std::move(py2DVec);;
            }
        }else{
            createPyVector(obj,pyObject,tableFlag,poption);
        }
    } else if (form == DF_TABLE) {
        TableSP ddbTbl = obj;
        if(poption->table2List==false){
            size_t columnSize = ddbTbl->columns();
            using namespace py::literals;
            py::array first = toPython(ddbTbl->getColumn(0), true);
            auto colName = py::list();
            colName.append(py::str(ddbTbl->getColumnName(0)));
            py::object dataframe = Preserved::pandas_.attr("DataFrame")(first, "columns"_a = colName);
            for (size_t i = 1; i < columnSize; ++i) {
                dataframe[ddbTbl->getColumnName(i).data()] = toPython(ddbTbl->getColumn(i), true, poption);
            }
            pyObject=std::move(dataframe);
        }else{
            size_t columnSize = ddbTbl->columns();
            py::list pyList(columnSize);
            for (size_t i = 0; i < columnSize; ++i) {
                py::object temp = toPython(ddbTbl->getColumn(i),true,poption);
                pyList[i] = temp;
            }
            pyObject=std::move(pyList);
        }
    } else if (form == DF_SCALAR) {
        switch (type) {
            case DT_VOID:
                pyObject=std::move(py::none());
                break;
            case DT_BOOL:
                pyObject=std::move(py::bool_(obj->getBool()));
                break;
            case DT_CHAR:
            case DT_SHORT:
            case DT_INT:
            case DT_LONG:
                pyObject=std::move(py::int_(obj->getLong()));
                break;
            case DT_DATE:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "D"));
                break;
            case DT_MONTH:
                pyObject=std::move(Preserved::datetime64_(obj->getLong() - 23640, "M"));
                break;    // ddb starts from 0000.0M
            case DT_TIME:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "ms"));
                break;
            case DT_MINUTE:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "m"));
                break;
            case DT_SECOND:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "s"));
                break;
            case DT_DATETIME:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "s"));
                break;
            case DT_DATEHOUR:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "h"));
                break;
            case DT_TIMESTAMP:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "ms"));
                break;
            case DT_NANOTIME:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "ns"));
                break;
            case DT_NANOTIMESTAMP:
                pyObject=std::move(Preserved::datetime64_(obj->getLong(), "ns"));
                break;
            case DT_FLOAT:
            case DT_DOUBLE:
                pyObject=std::move(py::float_(obj->getDouble()));
                break;
            case DT_IP:
            case DT_UUID:
            case DT_INT128:
            case DT_SYMBOL:
            case DT_BLOB:
            case DT_STRING:
                pyObject=std::move(py::str(obj->getString()));
                break;
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
            for (int i = 0; i < keys->size(); ++i) { pyDict[keys->getString(i).data()] = toPython(values->get(i),false,poption); }
        } else {
            for (int i = 0; i < keys->size(); ++i) { pyDict[py::int_(keys->getLong(i))] = toPython(values->get(i),false,poption); }
        }
        pyObject=std::move(pyDict);
    } else if (form == DF_MATRIX) {
        ConstantSP ddbMat = obj;
        size_t rows = ddbMat->rows();
        size_t cols = ddbMat->columns();
        // FIXME: currently only support numerical matrix
        if (ddbMat->getCategory() == MIXED) { throw RuntimeException("currently only support single typed matrix"); }
        ddbMat->setForm(DF_VECTOR);
        py::array pyMat = toPython(ddbMat,false,poption);
        py::object pyMatRowLabel = toPython(ddbMat->getRowLabel(),false,poption);
        py::object pyMatColLabel = toPython(ddbMat->getColumnLabel(),false,poption);
        pyMat.resize({cols, rows});
        pyMat = pyMat.attr("transpose")();
        py::list pyMatList;
        pyMatList.append(pyMat);
        pyMatList.append(pyMatRowLabel);
        pyMatList.append(pyMatColLabel);
        pyObject=std::move(pyMatList);
    } else if (form == DF_PAIR) {
        VectorSP ddbPair = obj;
        py::list pyPair;
        for (int i = 0; i < ddbPair->size(); ++i) { pyPair.append(toPython(ddbPair->get(i),false,poption)); }
        pyObject=std::move(pyPair);
    } else if (form == DF_SET) {
        VectorSP ddbSet = obj->keys();
        py::set pySet;
        for (int i = 0; i < ddbSet->size(); ++i) { pySet.add(toPython(ddbSet->get(i),false,poption)); }
        pyObject=std::move(pySet);
    } else {
        throw RuntimeException("the form is not supported! ");
    }
    DLOG("toPython", Util::getDataTypeString(obj->getType()).data(),Util::getDataFormString(obj->getForm()).data(),obj->size(), tableFlag,"}");
    return pyObject;
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
        return DT_DATETIME;
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

static inline bool isValueNull(long long value){
    return value == npLongNan_;
}

static inline bool isValueNull(double value){
    uint64_t r;
    memcpy(&r, &value, sizeof(r));
    return r == npDoubleNan_;
}

template <typename T>
void processData(T *psrcData, int size, std::function<void(T *, int)> f) {
    int bufsize = std::min(1024, size);
    T buf[bufsize];
    int startIndex = 0, len;
    while(startIndex < size){
        len = std::min(size-startIndex, bufsize);
        memcpy(buf, psrcData+startIndex, sizeof(T)*len);
        f(buf,len);
        startIndex += len;
    }
}

void AddVectorData(VectorSP &ddbVec, py::array &pyVec, DATA_TYPE type,int size) {
    //RECORDTIME(Util::getDataTypeString(type)+"AddVectorData");
    DLOG("{ AddVectorData",Util::getDataTypeString(type),size,"start");
    switch (type) {
        case DT_BOOL: {
            pyVec = pyVec.attr("astype")("int8");
            ddbVec->appendBool((char*)pyVec.data(), size);
            break;
        }
        case DT_CHAR: {
            pyVec = pyVec.attr("astype")("int8");
            ddbVec->appendChar((char*)pyVec.data(), size);
            break;
        }
        case DT_SHORT: {
            //append<short>(pyVec, size, [&](short *buf, int size) { ddbVec->appendShort(buf, size); });
            pyVec = pyVec.attr("astype")("int16");
            ddbVec->appendShort((short*)pyVec.data(),size);
            break;
        }
        case DT_INT: {
            //append<int>(pyVec, size, [&](int *buf, int size) { ddbVec->appendInt(buf, size); });
            pyVec = pyVec.attr("astype")("int32");
            ddbVec->appendInt((int*)pyVec.data(),size);
            break;
        }
        case DT_MONTH: {
            pyVec = pyVec.attr("astype")("int64");
            processData<long long>((long long*)pyVec.data(), size, [&](long long *buf, int size) {
                for(int i = 0; i < size; ++i)
                    buf[i] += 23640;
                ddbVec->appendLong(buf, size);
            });
            break;
        }
        case DT_DATE:
        case DT_TIME:
        case DT_MINUTE:
        case DT_SECOND:
        case DT_DATETIME:
        case DT_DATEHOUR:
        case DT_TIMESTAMP:
        case DT_NANOTIME:
        case DT_NANOTIMESTAMP:
        case DT_LONG: {
            //append<long long>(pyVec, size, [&](long long *buf, int size) { ddbVec->appendLong(buf, size); });
            pyVec = pyVec.attr("astype")("int64");
            ddbVec->appendLong((long long*)pyVec.data(),size);
            break;
        }
        case DT_FLOAT: {
            //append<float>(pyVec, size, [&](float *buf, int size) { ddbVec->appendFloat(buf, size); });
            pyVec = pyVec.attr("astype")("float32");
            bool hasnull=false;
            processData<float>((float*)pyVec.data(), size, [&](float *buf, int size) {
                for(int i=0; i < size;i++){
                    if(buf[i]==NAN) {
                        buf[i] = FLT_NMIN;
                        hasnull=true;
                    }
                }
                ddbVec->appendFloat(buf, size);
            });
            ddbVec->setNullFlag(hasnull);
            break;
        }
        case DT_DOUBLE: {
            pyVec = pyVec.attr("astype")("float64");
            bool hasnull=false;
            processData<double>((double*)pyVec.data(), size, [&](double *buf, int size) {
                for(int i=0; i < size;i++){
                    if(buf[i] == NAN) {
                        buf[i] = DBL_NMIN;
                        hasnull=true;
                    }
                }
                ddbVec->appendDouble(buf, size);
            });
            ddbVec->setNullFlag(hasnull);
            break;
        }
        case DT_IP:
        case DT_UUID:
        case DT_INT128:
        case DT_SYMBOL:
        case DT_BLOB:
        case DT_STRING:
        {
            vector<std::string> strs;
            strs.reserve(size);
            char *buffer;
            ssize_t length;
            for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {
                PyObject *utf8obj = PyUnicode_AsUTF8String(it->ptr());
                if (utf8obj != NULL){
                    PyBytes_AsStringAndSize(utf8obj, &buffer, &length);
                    std::string str;
                    str.assign(buffer, (size_t)length);
                    strs.emplace_back(std::move(str));
                    Py_DECREF(utf8obj);
                }else{
                    DLOG("can't get utf-8 object from string.");
                }
                //strs.emplace_back(py::reinterpret_borrow<py::str>(*it).cast<std::string>());
            }
            ddbVec->appendString(strs.data(), strs.size());
            break;
        }
        case DT_ANY: {
            // extra check (determine string vector or any vector)
            for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {
                ConstantSP item = DdbPythonUtil::toDolphinDB(py::reinterpret_borrow<py::object>(*it));
                ddbVec->append(item);
            }
            break;
        }
        default: {
            throw RuntimeException("type error in numpy: " + Util::getDataTypeString(type));
        }
    }
    DLOG("AddVectorData",Util::getDataTypeString(type),size,"end }");
}

void DdbPythonUtil::toDolphinDBScalar(const py::object *obj, const DATA_TYPE *type, int size, vector<ConstantSP> &result){
    for(int i = 0; i < size; i++){
        result.push_back(_toDolphinDBScalar(obj[i], type[i]));
    }
}

void DdbPythonUtil::toDolphinDBScalar(const py::object *obj, int size, DATA_TYPE type, vector<ConstantSP> &result){
    for(int i = 0; i < size; i++){
        result.push_back(_toDolphinDBScalar(obj[i], type));
    }
}

inline ConstantSP DdbPythonUtil::_toDolphinDBScalar(const py::object &obj, DATA_TYPE typeIndicator) {
    //RECORDTIME("toDolphinDBScalar");
    //DLOG("toDolphinDBScalar.start.");
    if(typeIndicator >= ARRAY_TYPE_BASE){//Element in ArrayVector, it's a vector
        DATA_TYPE eleType = (DATA_TYPE)(typeIndicator - ARRAY_TYPE_BASE);
        ConstantSP dataVector;
        if(createVectorMatrix(obj, eleType, dataVector,ANY_ARRAY_VECTOR_OPTION::AAV_DISABLE) == false){
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
    } else if(py::isinstance(obj, Preserved::pdNaT)){
        return Util::createNullConstant(typeIndicator);
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
            constobj=Util::createDateTime(value*60);
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

DATA_TYPE toDolphinDBDataType(const py::object &obj, bool &isnull) {
    isnull=false;
    if (py::isinstance(obj, Preserved::pynone_)) {
        isnull=true;
        return DATA_TYPE::DT_OBJECT;
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
    }else if(py::isinstance(obj, Preserved::pdNaT)){
        isnull=true;
        return DATA_TYPE::DT_NANOTIMESTAMP;
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
            return DATA_TYPE::DT_DATETIME;
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
            DLOG("unsupported python data type numpy.datetime64 [%s].",py::str(obj.get_type()).cast<std::string>().data());
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
        DLOG("unsupported python data type ",py::str(obj.get_type()).cast<std::string>().data());
        return DATA_TYPE::DT_OBJECT;
    }
}

inline bool isObjArray(py::object obj){
    return py::isinstance(obj, Preserved::pytuple_)||
            py::isinstance(obj, Preserved::pylist_)||
            py::isinstance(obj, Preserved::nparray_)||
            py::isinstance(obj, Preserved::pdseries_);
}

//if pdataaddress is null, no children
bool getVectorChildren(py::object &obj, DATA_TYPE &type, vector<py::object> &children, size_t &rows, size_t &cols){
    if(py::isinstance(obj, Preserved::pytuple_)){
        DLOG("pytuple_.");
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
        DLOG("pylist_.");
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
        DLOG("nparray_series_dim", dim);
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
        if(type == DT_OBJECT){
            type = numpyToDolphinDBType(pyVec);
            //DLOG("VectorNumpy_numpyToDolphinDBType: %s.",Util::getDataTypeString(type).data());
        }
        children.resize(pyVec.size());
        int index=0;
        for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {//check if child is array
            children[index++]=py::reinterpret_borrow<py::object>(*it);
        }
        //DLOG("nparrayseries_ end.");
    }else{
        return false;
    }
    return true;
}

bool checkType(const py::object &obj,DATA_TYPE &type,DATA_TYPE &nullType,DdbPythonUtil::ANY_ARRAY_VECTOR_OPTION option, bool &isArrayVector){
    if(isObjArray(obj)){//has child vector
        if(option == DdbPythonUtil::AAV_ANYVECTOR){//pref any vector
            type = DATA_TYPE::DT_ANY;
        }else if(option == DdbPythonUtil::AAV_ARRAYVECTOR){//pref array vector
            isArrayVector =true;
        }else if(option == DdbPythonUtil::AAV_DISABLE){//disable child vector
            throw RuntimeException("unexpected vector form in object.");
        }
        return true;
    }
    if(py::isinstance(obj, Preserved::pynone_))//null can be any type, ignore it
        return false;
    bool isnull;
    DATA_TYPE curType = toDolphinDBDataType(obj,isnull);
    DLOG("getChildrenType",Util::getDataTypeString(curType).data(),"isnull",isnull);
    if(isnull){
        if(curType != DT_OBJECT)
            nullType = curType;
        //keep type last value.
        return false;
    }
    if(curType == DT_OBJECT){
        DLOG("toAny for Object type.");
        type = DATA_TYPE::DT_ANY;
        return true;
    }
    if(type == DT_OBJECT){//type is default value, set it
        type = curType;
        return false;
    }
    if(type != curType){//two types, set any
        DLOG("toAny for",Util::getDataTypeString(type).data(),"!=",Util::getDataTypeString(curType).data());
        type = DATA_TYPE::DT_ANY;
        return true;
    }
    return false;
}

DATA_TYPE getChildrenType(const vector<py::object> &children, DdbPythonUtil::ANY_ARRAY_VECTOR_OPTION option, bool &isArrayVector){
    DATA_TYPE type = DT_OBJECT;
    DATA_TYPE nullType = DT_OBJECT;
    for(auto &one : children){
        if(checkType(one,type,nullType,option,isArrayVector))
            break;
    }
    if(type == DT_OBJECT){//Is all none???
        if(nullType != DT_OBJECT){//Set null object type
            type = nullType;
            DLOG("all null, set to last null type",Util::getDataTypeString(nullType).data());
        }
    }
    return type;
}

bool DdbPythonUtil::createVectorMatrix(py::object obj, DATA_TYPE typeIndicator, ConstantSP &ddbResult, ANY_ARRAY_VECTOR_OPTION option){
    //RECORDTIME("createVectorMatrix");
    vector<py::object> children;
    size_t rows, cols;
    bool isArrayVector = (typeIndicator >= ARRAY_TYPE_BASE);
    DATA_TYPE type = typeIndicator;
    if(isArrayVector==false&&
        (py::isinstance(obj, Preserved::nparray_)
            ||py::isinstance(obj, Preserved::pdseries_))){
        //numpy array
        py::array pyVec = obj;
        int dim = pyVec.ndim();
        if(dim == 1){
            DLOG("nparray_series_dim", dim);
            if(type == DT_OBJECT){
                type = numpyToDolphinDBType(pyVec);
                if(type == DT_OBJECT){
                    DATA_TYPE nullType = DT_OBJECT;
                    py::object obj;
                    for (auto it = pyVec.begin(); it != pyVec.end(); ++it) {
                        obj=py::reinterpret_borrow<py::object>(*it);
                        if(checkType(obj,type,nullType,option,isArrayVector))
                            break;
                        if(type != DT_OBJECT && option == ANY_ARRAY_VECTOR_OPTION::AAV_ARRAYVECTOR){
                            break;
                        }
                    }
                    if(isArrayVector==false){
                        if(type == DT_OBJECT){//Is all none???
                            if(nullType != DT_OBJECT){//Set null object type
                                type = nullType;
                                DLOG("all null, set to last null type",Util::getDataTypeString(nullType).data());
                            }
                        }
                    }
                }
            }
            if(isArrayVector==false && type != DT_OBJECT){
                int size = pyVec.size();
                VectorSP ddbVec;
                ddbVec = Util::createVector(type, 0, size);
                AddVectorData(ddbVec,pyVec,type,size);
                ddbResult=ddbVec;
                return true;
            }
        }
    }
    if(getVectorChildren(obj, type, children, rows, cols) == false){
        if(isArrayVector){
            throw RuntimeException("unexpected array vector object.");
        }
        return false;
    }
    if(isArrayVector == false && type == DT_OBJECT){//unknow type, or if it is arrayvector in dataframe/np.array
        type = getChildrenType(children,option,isArrayVector);
        if(isArrayVector == false){
            if(type == DT_OBJECT){//Is all none???
                type = DT_DOUBLE;
            }
        }
    }
    DLOG("{ createVectorMatrix type",Util::getDataTypeString(type).data(),rows,cols,children.size(),"isarrayvector",isArrayVector,option);
    if(isArrayVector){
        //RECORDTIME("createVectorMatrix_arrayvector");
        if(rows > 1){
            throw RuntimeException("unpexpected matrix for array vector object.");
        }
        DATA_TYPE eleType;
        Vector *pArrayVector = NULL;
        VectorSP pchildVector;
        vector<py::object> grandsons;
        size_t grandsonRows, grandsonCols;
        size_t startChildIndex = 0;
        size_t childSize = children.size();
        if(type == DT_OBJECT){//unknow type
            vector<int> noneRowCounts;
            bool tmpIsArrayVector;
            noneRowCounts.reserve(childSize);
            for(; startChildIndex < childSize; startChildIndex++){
                if(getVectorChildren(children[startChildIndex], type, grandsons, grandsonRows, grandsonCols) == false){
                    throw RuntimeException("invalid array vector object.");
                }
                if(grandsonRows > 1){
                    throw RuntimeException("unpexpected matrix for array vector object.");
                }
                if(type == DT_ANY){
                    throw RuntimeException("unpexpected any type for array vector object.");
                }
                if(type == DT_OBJECT){// unknow type
                    type = getChildrenType(grandsons,ANY_ARRAY_VECTOR_OPTION::AAV_DISABLE,tmpIsArrayVector);
                    if(type == DT_OBJECT){//Is all none? save none object size
                        noneRowCounts.push_back(grandsons.size());
                        continue;
                    }
                }
                DLOG("arrayvector type determined",Util::getDataTypeString(type).data());
                eleType = type;
                type = (DATA_TYPE)(eleType + ARRAY_TYPE_BASE);
                pArrayVector = Util::createArrayVector(type, 0, children.size());
                ddbResult = pArrayVector;

                if(noneRowCounts.size() > 0){//has all none object
                    DLOG("add void %d.",noneRowCounts.size());
                    for(auto count : noneRowCounts){
                        pchildVector = Util::createVector(eleType, 0, count);
                        for( int i = 0; i < count; i ++){
                            pchildVector->append(Constant::void_);
                        }
                        pArrayVector->append(pchildVector);
                    }
                }
                break;
            }
            if(pArrayVector == NULL){
                throw RuntimeException("unexpected all none object in array vector.");
            }
        }else{
            eleType = (DATA_TYPE)(type - ARRAY_TYPE_BASE);
            pArrayVector = Util::createArrayVector(type, 0, children.size());
            ddbResult = pArrayVector;
        }
        DATA_TYPE tmpType = type;
        VectorSP panyVector;
        for(; startChildIndex < childSize; startChildIndex++){
            if(getVectorChildren(children[startChildIndex], tmpType, grandsons, grandsonRows, grandsonCols) == false){
                throw RuntimeException("invalid array vector object.");
            }
            if(grandsonRows > 1){
                throw RuntimeException("unpexpected matrix for array vector object.");
            }
            pchildVector = Util::createVector(eleType, 0, grandsons.size());
            vector<ConstantSP> list;
            list.reserve(grandsons.size());
            toDolphinDBScalar(grandsons.data(),grandsons.size(), eleType, list);
            for(size_t i=0;i<list.size();i++){
                pchildVector->append(list[i]);
            }
            panyVector=Util::createVector(DT_ANY,0,1);
            panyVector->append(pchildVector);
            pArrayVector->append(panyVector);
        }
    }else if(rows <= 1){//vector
        //DLOG("createVector_%s.",Util::getDataTypeString(type).data());
        //RECORDTIME("createVectorMatrix_any");
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
        ddbResult=ddbVec;
    }else{//matrix
        //RECORDTIME("createVectorMatrix_matrix");
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
        ddbResult=ddbMat;
    }
    DLOG("createVectorMatrix type",Util::getDataTypeString(type).data(),rows,cols,children.size(),"}");
    return true;
}

ConstantSP DdbPythonUtil::toDolphinDB(py::object obj, DATA_FORM formIndicator, DATA_TYPE typeIndicator) {
    //RECORDTIME("toDolphinDB");
    DLOG("{ toDolphinDB start",Util::getDataTypeString(typeIndicator).data(),Util::getDataFormString(formIndicator).data());
    ConstantSP ddbConst;
    if(isObjArray(obj) == false || createVectorMatrix(obj, typeIndicator, ddbConst, AAV_ANYVECTOR) == false){//it's not vector
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
            DATA_TYPE type;
            for (size_t i = 0; i < columnSize; ++i) {
                DLOG("pddataframe column" , i);
                py::object boject;
                if (typeIndicators.contains(columnNames[i].data())) {
                    type = static_cast<DATA_TYPE>(typeIndicators[columnNames[i].data()].cast<int>());
                }else{
                    type = DT_OBJECT;
                }
                ConstantSP ddbColumn;
                if(createVectorMatrix(py::array(dataframe[columnNames[i].data()]), type, ddbColumn, ANY_ARRAY_VECTOR_OPTION::AAV_ARRAYVECTOR) == false){
                    throw RuntimeException("DolphinDB only support vector as column.");
                }
                columns.emplace_back(ddbColumn);
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
                    if(isnull){//set null default type: double
                        typeIndicator=DT_DOUBLE;
                    }else{
                        throw RuntimeException("DolphinDB doesn't support Python data " + py::str(obj.get_type()).cast<std::string>());
                    }
                }
            }
            //DLOG("toDolphinDB.toDolphinDBScalar.start.");
            DLOG("toDolphindb_toDolphinDBScalar",Util::getDataTypeString(typeIndicator).data());
            ddbConst=_toDolphinDBScalar(obj, typeIndicator);
        }
    }
    //DLOG("toDolphinDB.end.\n");
    DLOG("toDolphinDB", Util::getDataTypeString(ddbConst->getType()).data(),Util::getDataFormString(ddbConst->getForm()).data(),ddbConst->size(),"}");
    return ddbConst;
}

py::object DdbPythonUtil::loadPickleFile(const std::string &filepath){
    py::dict statusDict;
    string shortFilePath=filepath+"_s";
    FILE *pf;
    pf=fopen(shortFilePath.data(),"rb");
    if(pf==NULL){
        pf=fopen(filepath.data(),"rb");
        if(pf==NULL){
            statusDict["errorCode"]=-1;
            statusDict["errorInfo"]=filepath+" can't open.";
            return statusDict;
        }
        FILE *pfwrite=fopen(shortFilePath.data(),"wb");
        if(pfwrite==NULL){
            statusDict["errorCode"]=-1;
            statusDict["errorInfo"]=shortFilePath+" can't open.";
            return statusDict;
        }
        char buf[4096];
        int readlen;
        {
            char tmp;
            char header=0x80;
            int version;
            while(fread(&tmp,1,1,pf)==1){
                while(tmp==header){
                    if(fread(&tmp,1,1,pf)!=1)
                        break;
                    version=(unsigned char)tmp;
                    DLOG(version);
                    if(version>=0&&version<=5){
                        fwrite(&header,1,1,pfwrite);
                        fwrite(&tmp,1,1,pfwrite);
                        goto nextread;
                    }
                }
            }
        }
    nextread:
        while((readlen=fread(buf,1,4096,pf))>0){
            fwrite(buf,1,readlen,pfwrite);
        }
        fclose(pf);
        fclose(pfwrite);
        pf=fopen(shortFilePath.data(),"rb");
    }
    if(pf==NULL){
        statusDict["errorCode"]=-1;
        statusDict["errorInfo"]=filepath+" can't open.";
        return statusDict;
    }
    DataInputStreamSP dis=new DataInputStream(pf);
    std::unique_ptr<PickleUnmarshall> unmarshall(new PickleUnmarshall(dis));
    IO_ERR ret;
    short flag=0;
    if (!unmarshall->start(flag, true, ret)) {
        unmarshall->reset();
        statusDict["errorCode"]=(int)ret;
        statusDict["errorInfo"]="unmarshall failed";
        return statusDict;
    }
    PyObject * result = unmarshall->getPyObj();
    unmarshall->reset();
    py::object res = py::handle(result).cast<py::object>();
    res.dec_ref();
    return res;
}

PytoDdbRowPool::PytoDdbRowPool(MultithreadedTableWriter &writer)
                                : writer_(writer)
                                ,exitWhenEmpty_(false)
                                ,pGilRelease_(NULL)
                                ,convertingCount_(0)
                                {
    idle_.release();
    thread_=new Thread(new ConvertExecutor(*this));
    thread_->start();
}

PytoDdbRowPool::~PytoDdbRowPool(){
    DLOG("~PytoDdbRowPool",rows_.size(),failedRows_.size());
    if(!rows_.empty()||!failedRows_.empty()){
        ProtectGil protectGil;
        while(!rows_.empty()){
            delete rows_.front();
            rows_.pop();
        }
        while(!failedRows_.empty()){
            delete failedRows_.front();
            failedRows_.pop();
        }
    }
}

void PytoDdbRowPool::startExit(){
    DLOG("startExit with",rows_.size(),failedRows_.size());
    pGilRelease_=new py::gil_scoped_release;

    exitWhenEmpty_ = true;
    nonempty_.set();
    thread_->join();
}

void PytoDdbRowPool::endExit(){
    DLOG("endExit with",rows_.size(),failedRows_.size());
    delete pGilRelease_;
    pGilRelease_ = NULL;
}

void PytoDdbRowPool::convertLoop(){
    vector<std::vector<py::object>*> convertRows;
    while(writer_.hasError_ == false){
        nonempty_.wait();
        SemLock idleLock(idle_);
        idleLock.acquire();
        {
            RECORDTIME("rowPool:ConvertExecutor");
            LockGuard<Mutex> LockGuard(&mutex_);
            size_t size = rows_.size();
            if(size < 1){
                if(exitWhenEmpty_)
                    break;
                nonempty_.reset();
                continue;
            }
			if (size > 65535)
				size = 65535;
			convertRows.reserve(size);
            while(!rows_.empty()){
                convertRows.push_back(rows_.front());
                rows_.pop();
                if(convertRows.size() >= size)
                    break;
            }
            convertingCount_=convertRows.size();
        }
        {
            DLOG("convert start ",convertRows.size(),"/",rows_.size());
            vector<vector<ConstantSP>*> insertRows;
            insertRows.reserve(convertRows.size());
            vector<ConstantSP> *pDdbRow = NULL;
            try
            {
                ProtectGil protectGil;
                const DATA_TYPE *pcolType = writer_.getColType();
                int i, size;
                for (auto &prow : convertRows)
                {
                    pDdbRow = new vector<ConstantSP>;
                    size = prow->size();
                    for (i = 0; i < size; i++)
                    {
                        pDdbRow->push_back(DdbPythonUtil::_toDolphinDBScalar(prow->at(i), pcolType[i]));
                    }
                    insertRows.push_back(pDdbRow);
                    pDdbRow = NULL;
                    delete prow; // must delete it in GIL lock
                }
            }catch (RuntimeException &e){
                writer_.setError(ErrorCodeInfo::EC_InvalidObject, std::string("Data conversion error: ") + e.what());
                delete pDdbRow;
            }
            if(!insertRows.empty()){
                writer_.insertUnwrittenData(insertRows);
            }
            if (insertRows.size() != convertRows.size()){ // has error, rows left some
                LockGuard<Mutex> LockGuard(&mutex_);
                for(size_t i = insertRows.size(); i < convertRows.size(); i++){
                    failedRows_.push(convertRows[i]);
                }
            }
            DLOG("convert end ",insertRows.size(),failedRows_.size(),"/",rows_.size());
            convertingCount_ = 0;
            convertRows.clear();
        }
    }
}

void PytoDdbRowPool::getStatus(MultithreadedTableWriter::Status &status){
    py::gil_scoped_release release;
    writer_.getStatus(status);
    MultithreadedTableWriter::ThreadStatus threadStatus;
    LockGuard<Mutex> guard(&mutex_);
    status.unsentRows += rows_.size() + convertingCount_;
    status.sendFailedRows += failedRows_.size();
    
    threadStatus.unsentRows += rows_.size() + convertingCount_;
    threadStatus.sendFailedRows += failedRows_.size();
    
    status.threadStatus.insert(status.threadStatus.begin(),threadStatus);
}
void PytoDdbRowPool::getUnwrittenData(vector<vector<py::object>*> &pyData,vector<vector<ConstantSP>*> &ddbData){
    py::gil_scoped_release release;
    writer_.getUnwrittenData(ddbData);
    {
        SemLock idleLock(idle_);
        idleLock.acquire();
        LockGuard<Mutex> LockGuard(&mutex_);
        while(!failedRows_.empty()){
            pyData.push_back(failedRows_.front());
            failedRows_.pop();
        }
        while(!rows_.empty()){
            pyData.push_back(rows_.front());
            rows_.pop();
        }
    }
}

}