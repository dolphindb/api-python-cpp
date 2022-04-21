#ifndef DdbUtil_H_
#define DdbUtil_H_
#include "DolphinDB.h"
#include "Concurrent.h"
#include "MultithreadedTableWriter.h"
#include "pybind11/numpy.h"
#include "pybind11/embed.h"

#ifdef _MSC_VER
	#ifdef _USRDLL	
		#define EXPORT_DECL _declspec(dllexport)
	#else
		#define EXPORT_DECL __declspec(dllimport)
	#endif
#else
	#define EXPORT_DECL 
#endif

namespace dolphindb {

//#define DLOG DLogger::Info
struct Preserved {
    // instantiation only once for frequently use

    // modules and methods
    static const py::object numpy_;         // module
    //static const py::object isnan_;         // func
    //static const py::object sum_;           // func
    static const py::object datetime64_;    // type, equal to np.datetime64
    static const py::object pandas_;        // module

    // pandas types (use py::isinstance)
    static const py::object pdseries_;
    static const py::object pddataframe_;
    static const py::object pdNaT;

    // numpy dtypes (instances of dtypes, use equal)
    static const py::object nparray_;
    static const py::object npbool_;
    static const py::object npint8_;
    static const py::object npint16_;
    static const py::object npint32_;
    static const py::object npint64_;
    static const py::object npfloat32_;
    static const py::object npfloat64_;

    static py::object npdatetime64M_();
    static py::object npdatetime64D_();
    static py::object npdatetime64m_();
    static py::object npdatetime64s_();
    static py::object npdatetime64h_();
    static py::object npdatetime64ms_();
    static py::object npdatetime64us_();
    static py::object npdatetime64ns_();
    static py::object npdatetime64_();
    static const py::object npobject_;
    
    static const py::object pynone_;
    static const py::object pybool_;
    static const py::object pyint_;
    static const py::object pyfloat_;
    static const py::object pystr_;
    static const py::object pybytes_;
    static const py::object pyset_;
    static const py::object pytuple_;
    static const py::object pylist_;
    static const py::object pydict_;
    static py::object getDType(const char *pname){
        py::dtype type(pname);
        //py::object dtname=py::getattr(type, "name");
        //std::string name=py::str(dtname);
        //printf("DType: %s.",name.data());
        return type;
    }
    static py::object getType(py::object obj){
        //std::string text=py::str(obj.get_type());
        //printf("Type: %s.",text.data());
        return py::reinterpret_borrow<py::object>(obj.get_type());
    }
};

class EXPORT_DECL DdbPythonUtil{
public:
    enum ANY_ARRAY_VECTOR_OPTION{
        AAV_ANYVECTOR,//create any vector to contain child vector
        AAV_ARRAYVECTOR,//create array vector
        AAV_DISABLE,// disable vector to contain child vector
    };
    struct ToPythonOption{
        bool table2List;//if object is table, false: convert to pandas, true: convert to list
        ToPythonOption(){
            table2List = false;
        }
        ToPythonOption(bool table2Lista){
            table2List = table2Lista;
        }
    };

    static bool createVectorMatrix(py::object obj, DATA_TYPE typeIndicator, ConstantSP &ddbvector, ANY_ARRAY_VECTOR_OPTION option);
    static ConstantSP toDolphinDB(py::object obj, DATA_FORM formIndicator = DF_CHUNK, DATA_TYPE typeIndicator = DT_OBJECT);
    //support arrayvector element too.
    static void toDolphinDBScalar(const py::object *obj, const DATA_TYPE *type, int size, vector<ConstantSP> &result);
    static void toDolphinDBScalar(const py::object *obj, int size, DATA_TYPE type, vector<ConstantSP> &result);
    static py::object toPython(ConstantSP obj, bool tableFlag=false, const ToPythonOption *poption = NULL);
    static py::object loadPickleFile(const std::string &filepath);
    static void createPyVector(const ConstantSP &obj,py::object &pyObject,bool tableFlag,const ToPythonOption *poption);
protected:
    friend class PytoDdbRowPool;
    static inline ConstantSP _toDolphinDBScalar(const py::object &obj, DATA_TYPE typeIndicator);
};

class PytoDdbRowPool{
public:
    PytoDdbRowPool(MultithreadedTableWriter &writer);
    ~PytoDdbRowPool();
    void startExit();
    void endExit();
    bool isExit(){
        return exitWhenEmpty_ || writer_.hasError_;
    }
    bool add(vector<py::object> *prow){
        if(isExit())
            return false;
        LockGuard<Mutex> LockGuard(&mutex_);
        rows_.push(prow);
        nonempty_.set();
        return true;
    }
    void getStatus(MultithreadedTableWriter::Status &status);
    void getUnwrittenData(vector<vector<py::object>*> &pyData,vector<vector<ConstantSP>*> &ddbData);

    class ConvertExecutor : public Runnable
    {
    public:
        ConvertExecutor(PytoDdbRowPool &rowPool) : rowPool_(rowPool){};
        virtual void run(){ rowPool_.convertLoop(); }
    private:
        PytoDdbRowPool &rowPool_;
    };

private:
    friend class ConvertExecutor;
    void convertLoop();
    MultithreadedTableWriter &writer_;
    ThreadSP thread_;
    bool exitWhenEmpty_;
    Semaphore idle_;
    Signal nonempty_;
    Mutex mutex_;
    int convertingCount_;
	std::queue<vector<py::object>*> rows_;
    std::queue<vector<py::object>*> failedRows_;
    //when main thread call exit, new pGilRelease_ to release GIL, then run can exit
    py::gil_scoped_release *pGilRelease_;
};

}

#endif //DdbUtil_H_