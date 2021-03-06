//
// Created by jccai on 3/28/19.
//

#include <DolphinDB.h>
#include <Streaming.h>
#include <BatchTableWriter.h>
#include <MultithreadedTableWriter.h>
#include <DdbPythonUtil.h>
#include <Util.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace py = pybind11;
namespace ddb = dolphindb;
using std::cout;
using std::endl;

#if defined(__GNUC__) && __GNUC__ >= 4
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#define DLOG //ddb::DLogger::Info
#define RECORD_TIME //ddb::RecordTime _recordTime

class DBConnectionPoolImpl {
public:
    DBConnectionPoolImpl(const std::string& hostName, int port, int threadNum = 10, const std::string& userId = "", const std::string& password = "",
            bool loadBalance = false, bool highAvailability = false, bool reConnectFlag = true,bool compress = false, bool enablePickle = true)
            :dbConnectionPool_(hostName, port, threadNum, userId, password,loadBalance,highAvailability,reConnectFlag,compress,enablePickle),
                host_(hostName), port_(port), threadNum_(threadNum), userId_(userId), password_(password) {}
    ~DBConnectionPoolImpl() {}
    py::object run(const string &script, int taskId) {
        try {
            dbConnectionPool_.runPy(script, taskId, 4, 2);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        return py::none();
    }

    py::object run(const string &funcName, int taskId, const py::args &args) {
        vector<ddb::ConstantSP> ddbArgs;
        for (py::handle one : args) {
            py::object pyobj = py::reinterpret_borrow<py::object>(one);
            ddb::ConstantSP pcp = ddb::DdbPythonUtil::toDolphinDB(pyobj);
            ddbArgs.push_back(pcp);
        }
        try {
            dbConnectionPool_.runPy(funcName, ddbArgs, taskId);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in call: ") + ex.what()); }
        return py::none();
    }

    py::object run(const string &script, int taskId, const py::kwargs & kwargs) {
        bool clearMemory = false;
        if(kwargs.contains("clearMemory")){
            clearMemory = kwargs["clearMemory"].cast<bool>();
        }
        bool pickleTableToList = false;
        if(kwargs.contains("pickleTableToList")){
            pickleTableToList = kwargs["pickleTableToList"].cast<bool>();
        }
        try {
            dbConnectionPool_.runPy(script, taskId, 4, 2, 0, clearMemory, pickleTableToList);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        //ddb::DLogger::Info(script,"cost time\n",ddb::RecordTime::printAllTime());
        return py::none();
    }

    py::object run(const string &funcName, int taskId, const py::args &args, const py::kwargs &kwargs) {
        bool clearMemory = false;
        if(kwargs.contains("clearMemory")){
            clearMemory = kwargs["clearMemory"].cast<bool>();
        }
        bool pickleTableToList = false;
        if(kwargs.contains("pickleTableToList")){
            pickleTableToList = kwargs["pickleTableToList"].cast<bool>();
        }
        vector<ddb::ConstantSP> ddbArgs;
        for (auto it = args.begin(); it != args.end(); ++it) { ddbArgs.push_back(ddb::DdbPythonUtil::toDolphinDB(py::reinterpret_borrow<py::object>(*it))); }
        try {
            dbConnectionPool_.runPy(funcName, ddbArgs, taskId, 4, 2, 0, clearMemory,pickleTableToList);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in call: ") + ex.what()); }
        return py::none();
    }
    bool isFinished(int taskId) {
        bool isFinished;
        try {
            isFinished = dbConnectionPool_.isFinished(taskId);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        return isFinished;
    }
    py::object getData(int taskId) {
        py::object result;
        try {
            result = dbConnectionPool_.getPyData(taskId);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        return result;
    }
    void shutDown() {
        host_ = "";
        port_ = 0;
        userId_ = "";
        password_ = "";
        dbConnectionPool_.shutDown();
    }

    py::object getSessionId() {
        vector<string> sessionId = dbConnectionPool_.getSessionId();
        py::list ret;
        for(auto &id: sessionId){
            ret.append(py::str(id));
        }
        return ret;
    }

    ddb::DBConnectionPool& getPool() {
        return dbConnectionPool_;
    }
    
private:
    ddb::DBConnectionPool dbConnectionPool_;
    std::string host_;
    int port_;
    int threadNum_;
    std::string userId_;
    std::string password_;
};

class BlockReader{
public:
    BlockReader(ddb::BlockReaderSP reader): reader_(reader){
    }
    ~BlockReader(){
    }
    void skipAll() {
        reader_->skipAll();
    }
    py::bool_ hasNext(){
        return py::bool_(reader_->hasNext());
    }
    py::object read(){
        py::object ret;
        try{
            ret = ddb::DdbPythonUtil::toPython(reader_->read());
        }catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in read: ") + ex.what()); }
        return ret;
    }

private:
    ddb::BlockReaderSP reader_;
};

class PartitionedTableAppender{
public:
    PartitionedTableAppender(string dbUrl, string tableName, string partitionColName, DBConnectionPoolImpl& pool)
    :partitionedTableAppender_(dbUrl,tableName,partitionColName,pool.getPool()){}
    int append(py::object table){
        if(!py::isinstance(table, ddb::Preserved::pddataframe_))
            throw std::runtime_error(std::string("table must be a DataFrame!"));
        int insertRows;
        try {
            insertRows = partitionedTableAppender_.append(ddb::DdbPythonUtil::toDolphinDB(table));
        }catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in append: ") + ex.what()); }
        return insertRows;
    }
private:
    ddb::PartitionedTableAppender partitionedTableAppender_;
};


// FIXME: not thread safe
class SessionImpl {
public:
    SessionImpl(bool enableSSL=false, bool enableASYN=false, int keepAliveTime=7200, bool compress=false, bool enablePickle=true) : host_(), port_(-1), userId_(), password_(), encrypted_(true),
            dbConnection_(enableSSL,enableASYN, keepAliveTime, compress, enablePickle), nullValuePolicy_([](ddb::VectorSP) {}), subscriber_(nullptr),subscriberPool_(nullptr),keepAliveTime_(keepAliveTime) {}

    bool connect(const std::string &host, const int &port, const std::string &userId, const std::string &password, const std::string &startup = "", const bool &highAvailability = false,
                 const py::list &highAvailabilitySites = py::list(0), const int &keepAliveTime=30) {
        host_ = host;
        port_ = port;
        userId_ = userId;
        password_ = password;
        bool isSuccess = false;
        if(keepAliveTime > 0){
            dbConnection_.setKeepAliveTime(keepAliveTime);
        }
        try {
            vector<string> sites;
            for (py::handle o : highAvailabilitySites) { sites.emplace_back(py::cast<std::string>(o)); }
            isSuccess = dbConnection_.connect(host_, port_, userId_, password_, startup, highAvailability, sites);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in connect: ") + ex.what()); }
        return isSuccess;
    }

    void setInitScript(string script) {
        try {
            dbConnection_.setInitScript(script);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in connect: ") + ex.what()); }
    }

    string getInitScript() {
        try {
            return dbConnection_.getInitScript();
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in connect: ") + ex.what()); }
    }

    void login(const std::string &userId, const std::string &password, bool enableEncryption) {
        try {
           dbConnection_.login(userId, password, enableEncryption);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in login: ") + ex.what()); }
    }

    void close() {
        host_ = "";
        port_ = 0;
        userId_ = "";
        password_ = "";
        dbConnection_.close();
    }

    const string getSessionId(){
        return dbConnection_.getSessionId();
    }

    py::object loadPickleFile(const std::string &filepath){
        return ddb::DdbPythonUtil::loadPickleFile(filepath);
    }
    py::object upload(const py::dict &namedObjects) {
        vector<std::string> names;
        vector<ddb::ConstantSP> objs;
        for (auto it = namedObjects.begin(); it != namedObjects.end(); ++it) {
            if (!py::isinstance(it->first, ddb::Preserved::pystr_) && !py::isinstance(it->first, ddb::Preserved::pybytes_)) { throw std::runtime_error("non-string key in upload dictionary is not allowed"); }
            names.push_back(it->first.cast<std::string>());
            objs.push_back(ddb::DdbPythonUtil::toDolphinDB(py::reinterpret_borrow<py::object>(it->second)));
        }
        try {
            auto addr = dbConnection_.upload(names, objs);
            if (addr == NULL || addr->getType() == ddb::DT_VOID ||addr->isNothing()) {
                return py::int_(-1);
            } else if(addr->isScalar()){
                return py::int_(addr->getLong());
            } else {
                size_t size = addr->size();
                py::list pyAddr;
                for (size_t i = 0; i < size; ++i) { pyAddr.append(py::int_(addr->getLong(i))); }
                return pyAddr;
            }
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in upload: ") + ex.what()); }
    }

    py::object run(const string &script) {
        py::object result;
        try {
            //ddb::RecordTime::printAllTime();
            result = dbConnection_.runPy(script, 4, 2);
            DLOG(ddb::RecordTime::printAllTime());
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        return result;
    }

    py::object run(const string &funcName, const py::args &args) {
        //ddb::RecordTime::printAllTime();
        py::object result;
        try {
            vector<ddb::ConstantSP> ddbArgs;
            int index=0;
            for (py::handle one : args) {
                py::object pyobj = py::reinterpret_borrow<py::object>(one);
                ddb::ConstantSP pcp = ddb::DdbPythonUtil::toDolphinDB(pyobj);
                ddbArgs.push_back(pcp);
                index++;
            }
            result = dbConnection_.runPy(funcName, ddbArgs);
            DLOG(ddb::RecordTime::printAllTime());
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in call: ") + ex.what()); }
        return result;
    }

    py::object run(const string &script, const py::kwargs & kwargs) {
        bool clearMemory = false;
        if(kwargs.contains("clearMemory")){
            clearMemory = kwargs["clearMemory"].cast<bool>();
        }
        bool pickleTableToList = false;
        if(kwargs.contains("pickleTableToList")){
            pickleTableToList = kwargs["pickleTableToList"].cast<bool>();
        }
        py::object result;
        try {
            //ddb::RecordTime::printAllTime();
            result = dbConnection_.runPy(script, 4, 2, 0, clearMemory, pickleTableToList);
            DLOG(ddb::RecordTime::printAllTime());
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        return result;
    }

    py::object run(const string &funcName, const py::args &args, const py::kwargs &kwargs) {
        bool clearMemory = false;
        if(kwargs.contains("clearMemory")){
            clearMemory = kwargs["clearMemory"].cast<bool>();
        }
        bool pickleTableToList = false;
        if(kwargs.contains("pickleTableToList")){
            pickleTableToList = kwargs["pickleTableToList"].cast<bool>();
        }
        py::object result;
        //ddb::RecordTime::printAllTime();
        try {
            vector<ddb::ConstantSP> ddbArgs;
            for (auto it = args.begin(); it != args.end(); ++it) { ddbArgs.push_back(ddb::DdbPythonUtil::toDolphinDB(py::reinterpret_borrow<py::object>(*it))); }
            result = dbConnection_.runPy(funcName, ddbArgs, 4, 2, 0, clearMemory,pickleTableToList);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in call: ") + ex.what()); }
        DLOG(ddb::RecordTime::printAllTime());
        return result;
    }

    BlockReader runBlock(const string &script, const py::kwargs & kwargs) {
        int fetchSize = 0;
        bool clearMemory = false;
        if(kwargs.contains("clearMemory")){
            clearMemory = kwargs["clearMemory"].cast<bool>();
        }
        if(kwargs.contains("fetchSize")){
            fetchSize = kwargs["fetchSize"].cast<int>();
        }
        if(fetchSize < 8192) {
            throw std::runtime_error(std::string("<Exception> in run: fectchSize must be greater than 8192"));
        }
        ddb::ConstantSP result;
        try {
            result = dbConnection_.run(script, 4, 2, fetchSize, clearMemory);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in run: ") + ex.what()); }
        BlockReader blockReader(result);
        return blockReader;
    }

    void nullValueToZero() {
        nullValuePolicy_ = [](ddb::VectorSP vec) {
            if (!vec->hasNull() || vec->getCategory() == ddb::TEMPORAL || vec->getType() == ddb::DT_STRING || vec->getType() == ddb::DT_SYMBOL) {
                return;
            } else {
                ddb::ConstantSP val = ddb::Util::createConstant(ddb::DT_LONG);
                val->setLong(0);
                vec->nullFill(val);
                assert(!vec->hasNull());
            }
        };
    }

    void nullValueToNan() {
        nullValuePolicy_ = [](ddb::VectorSP) {};
    }

    void enableStreaming(int listeningPort, int threadcount) {
        if (subscriber_.isNull() && subscriberPool_.isNull()) {
            if(threadcount <= 1){
                subscriber_ = new ddb::ThreadedClient(listeningPort);
            }else{
                subscriberPool_ = new ddb::ThreadPooledClient(listeningPort, threadcount);
            }
            ddb::Util::sleep(100);
        } else {
            throw std::runtime_error("streaming is already enabled");
        }
    }


    void subscribe(const string &host, const int &port, py::object handler, const string &tableName, const string &actionName, const long long &offset, const bool &resub, py::array filter) {
        if (subscriber_.isNull() && subscriberPool_.isNull() ) { throw std::runtime_error("streaming is not enabled"); }
        ddb::LockGuard<ddb::Mutex> LockGuard(&subscriberMutex_);
        string topic = host + "/" + std::to_string(port) + "/" + tableName + "/" + actionName;
        if (topicThread_.find(topic) != topicThread_.end()) { throw std::runtime_error("subscription " + topic + " already exists"); }
        ddb::MessageHandler ddbHandler = [handler, this](ddb::Message msg) {
            // handle GIL
            py::gil_scoped_acquire acquire;
            size_t size = msg->size();
            py::list pyMsg;
            for (size_t i = 0; i < size; ++i) { pyMsg.append(ddb::DdbPythonUtil::toPython(msg->get(i))); }
            handler(pyMsg);
        };
        ddb::VectorSP ddbFilter = filter.size() ? ddb::DdbPythonUtil::toDolphinDB(filter) : nullptr;
        vector<ddb::ThreadSP> threads;
        if(subscriber_.isNull() == false){
            ddb::ThreadSP thread = subscriber_->subscribe(host, port, ddbHandler, tableName, actionName, offset, resub, ddbFilter);
            threads.push_back(thread);
        }else{
            threads = subscriberPool_->subscribe(host, port, ddbHandler, tableName, actionName, offset, resub, ddbFilter);
        }
        topicThread_[topic] = threads;
    }

      // FIXME: not thread safe
    void subscribeBatch(string &host,int port, py::object handler,string &tableName,string &actionName,long long offset, bool resub, py::array filter,
            const bool & msgAsTable, int batchSize, double throttle) {
        if (subscriber_.isNull()) {
            if(subscriberPool_.isNull()){
                throw std::runtime_error("streaming is not enabled");
            }else{
                throw std::runtime_error("Thread pool streaming doesn't support batch subscribe");
            }
        }
        ddb::LockGuard<ddb::Mutex> LockGuard(&subscriberMutex_);
        string topic = host + "/" + std::to_string(port) + "/" + tableName + "/" + actionName;
        if (topicThread_.find(topic) != topicThread_.end()) { throw std::runtime_error("subscription " + topic + " already exists"); }
        ddb::MessageBatchHandler ddbHandler = [handler, msgAsTable, this](std::vector<ddb::Message> msg) {
            // handle GIL
            py::gil_scoped_acquire acquire;
            size_t size = msg.size();   
            py::list pyMsg;
            for (size_t i = 0; i < size; ++i) {
                pyMsg.append(ddb::DdbPythonUtil::toPython(msg[i])); 
            }
            if(msgAsTable){
                py::object dataframe = ddb::Preserved::pandas_.attr("DataFrame")(pyMsg);
                handler(dataframe);
            }else {
                handler(pyMsg);
            }
        };
        ddb::VectorSP ddbFilter = filter.size() ? ddb::DdbPythonUtil::toDolphinDB(filter) : nullptr;
        vector<ddb::ThreadSP> threads;
        ddb::ThreadSP thread = subscriber_->subscribe(host, port, ddbHandler, tableName, actionName, offset, resub, ddbFilter, false, batchSize, throttle);
        threads.push_back(thread);
        topicThread_[topic] = threads;
    }

    // FIXME: not thread safe
    void unsubscribe(string host, int port, string tableName, string actionName) {
        if (subscriber_.isNull() && subscriberPool_.isNull()) { throw std::runtime_error("streaming is not enabled"); }
        ddb::LockGuard<ddb::Mutex> LockGuard(&subscriberMutex_);
        string topic = host + "/" + std::to_string(port) + "/" + tableName + "/" + actionName;
        if (topicThread_.find(topic) == topicThread_.end()) { throw std::runtime_error("subscription " + topic + " not exists"); }
        subscriber_->unsubscribe(host, port, tableName, actionName);
        vector<ddb::ThreadSP> &threads = topicThread_[topic];
        for(auto thread : threads){
            if(thread->isRunning()) {
                gcThread_.push_back(thread);
                auto it = std::remove_if(gcThread_.begin(), gcThread_.end(), [](const ddb::ThreadSP& th) {
                    return th->isComplete();
                });
                gcThread_.erase(it, gcThread_.end());
            }
        }
        topicThread_.erase(topic);
    }

    // FIXME: not thread safe
    py::list getSubscriptionTopics() {
        ddb::LockGuard<ddb::Mutex> LockGuard(&subscriberMutex_);
        py::list topics;
        for (auto &it : topicThread_) { topics.append(it.first); }
        return topics;
    }

    py::object hashBucket(const py::object& obj, int nBucket) {
        auto c = ddb::DdbPythonUtil::toDolphinDB(obj);
        const static auto errMsg = "Key must be integer, date/time, or string.";
        auto dt = c->getType();
        auto cat = ddb::Util::getCategory(dt);
        if (cat != ddb::INTEGRAL && cat != ddb::TEMPORAL && dt != ddb::DT_STRING) {
            throw std::runtime_error(errMsg);
        }

        if(c->isVector()) {
            int n = c->size();
            py::array h(py::dtype("int32"), n, {});
            c->getHash(0, n, nBucket, (int*)h.mutable_data());
            return h;
        } else {
            int h = c->getHash(nBucket);
            return py::int_(h);
        }
    }

    ~SessionImpl() {
        for (auto &it : topicThread_) {
            vector<std::string> args = ddb::Util::split(it.first, '/');
            try {
                unsubscribe(args[0], std::stoi(args[1]), args[2], args[3]);
            } catch (ddb::RuntimeException &ex) { std::cout << "exception occurred in SessionImpl destructor: " << ex.what() << std::endl; }
        }
        for (auto &it : topicThread_) {
            for(auto &thread : it.second){
                thread->join();
            }
        }
    }

    ddb::DBConnection& getConnection() {
        return dbConnection_;
    }
private:
    using policy = void (*)(ddb::VectorSP);

private:
    //static inline void SET_NPNAN(void *p, size_t len = 1) { std::fill((uint64_t *)p, ((uint64_t *)p) + len, 9221120237041090560LL); }
    //static inline void SET_DDBNAN(void *p, size_t len = 1) { std::fill((double *)p, ((double *)p) + len, ddb::DBL_NMIN); }
    //static inline bool IS_NPNAN(void *p) { return *(uint64_t *)p == 9221120237041090560LL; }


private:
    std::string host_;
    int port_;
    std::string userId_;
    std::string password_;
    bool encrypted_;
    ddb::DBConnection dbConnection_;
    policy nullValuePolicy_;

    ddb::SmartPointer<ddb::ThreadedClient> subscriber_;
    ddb::SmartPointer<ddb::ThreadPooledClient> subscriberPool_;
    std::unordered_map<string, std::vector<ddb::ThreadSP>> topicThread_;
    ddb::Mutex subscriberMutex_;
    std::vector<ddb::ThreadSP> gcThread_;
    int keepAliveTime_;
};

class AutoFitTableAppender{
public:
    AutoFitTableAppender(const std::string dbUrl, const std::string tableName, SessionImpl & session)
    : autoFitTableAppender_(dbUrl,tableName,session.getConnection()){}
    int append(py::object table){
        if(!py::isinstance(table, ddb::Preserved::pddataframe_))
            throw std::runtime_error(std::string("table must be a DataFrame!"));
        int insertRows;
        try {
            insertRows = autoFitTableAppender_.append(ddb::DdbPythonUtil::toDolphinDB(table));
        }catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in append: ") + ex.what()); }
        return insertRows;
    }
private:
    ddb::AutoFitTableAppender autoFitTableAppender_;
};

class BatchTableWriter{
public:
    BatchTableWriter(const std::string& hostName, int port, const std::string& userId, const std::string& password, bool acquireLock=true)
    : writer_(hostName, port, userId, password, acquireLock){}
    ~BatchTableWriter(){}
    void addTable(const string& dbName="", const string& tableName="", bool partitioned=true){
        try {
            writer_.addTable(dbName, tableName, partitioned);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in addTable: ") + ex.what()); }
    }
    py::object getStatus(const string& dbName, const string& tableName=""){
        try {
            std::tuple<int,bool,bool> tem = writer_.getStatus(dbName, tableName);
            py::list ret;
            ret.append(py::int_(std::get<0>(tem)));
            ret.append(py::bool_(std::get<1>(tem)));
            ret.append(py::bool_(std::get<2>(tem)));
            return ret;
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in getStatus: ") + ex.what()); }
    }
    py::object getAllStatus(){
        try {
            ddb::ConstantSP ret = writer_.getAllStatus();
            return ddb::DdbPythonUtil::toPython(ret);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in getAllStatus: ") + ex.what()); }
    }
    py::object getUnwrittenData(const string& dbName, const string& tableName=""){
        try {
            ddb::ConstantSP ret = writer_.getUnwrittenData(dbName, tableName);
            return ddb::DdbPythonUtil::toPython(ret);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in getUnwrittenData: ") + ex.what()); }
    }
    void removeTable(const string& dbName, const string& tableName=""){
        try {
            writer_.removeTable(dbName, tableName);
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in addTable: ") + ex.what()); }
    }
    void insert(const string& dbName, const string& tableName, const py::args &args){
        ddb::SmartPointer<vector<ddb::ConstantSP>> ddbArgs(new std::vector<ddb::ConstantSP>());
        int size = args.size();
        try {
            for (int i = 0; i < size; ++i){
                ddb::ConstantSP test = ddb::DdbPythonUtil::toDolphinDB(args[i]);
                ddbArgs->push_back(test);
            }
            writer_.insertRow(dbName, tableName, ddbArgs.get());
        } catch (ddb::RuntimeException &ex) { throw std::runtime_error(std::string("<Exception> in insert: ") + ex.what()); }
    }
private:
    ddb::BatchTableWriter writer_;
};

class MultithreadedTableWriter{
public:
    MultithreadedTableWriter(const std::string& host, int port, const std::string& userId, const std::string& password,
                                const string& dbPath, const string& tableName, bool useSSL, bool enableHighAvailability = false,
                                const py::list &highAvailabilitySites = py::list(0),
                                int batchSize = 1, float throttle = 0.01f,int threadCount = 1, const string& partitionCol ="",
                                const py::list &compressMethods = py::list(0)){
        try {
            writer_ = new ddb::MultithreadedTableWriter(host, port, userId, password,
                                dbPath, tableName, useSSL,
                                enableHighAvailability, pylist2Stringvector(highAvailabilitySites).get(),
                                batchSize,throttle,threadCount,partitionCol,
                                pylist2Compressvector(compressMethods).get());
        } catch (ddb::RuntimeException &ex) {
            throw std::runtime_error(std::string("<Exception> in init: ") + ex.what());
        }
    }
    ~MultithreadedTableWriter(){}
    void waitForThreadCompletion(){
        writer_->waitForThreadCompletion();
    }
    py::object getStatus(){
        py::dict pystatus;
        ddb::MultithreadedTableWriter::Status status;
        writer_->getPytoDdb()->getStatus(status);
        pystatus["isExiting"]=status.isExiting;
        pystatus["errorCode"]=status.errorCode;
        pystatus["errorInfo"]=status.errorInfo;
        pystatus["sentRows"]=status.sentRows;
        pystatus["unsentRows"]=status.unsentRows;
        pystatus["sendFailedRows"]=status.sendFailedRows;
        {
            py::list list(status.threadStatus.size());
            int index=0;
            for(auto thread : status.threadStatus){
                py::dict pythreadstatus;
                pythreadstatus["threadId"] = thread.threadId;
                pythreadstatus["sentRows"] = thread.sentRows;
                pythreadstatus["unsentRows"] = thread.unsentRows;
                pythreadstatus["sendFailedRows"] = thread.sendFailedRows;
                list[index++] = pythreadstatus;
            }
            pystatus["threadStatus"]=list;
        }
        return pystatus;
    }
    py::object getUnwrittenData(){
        try {
            std::vector<std::vector<ddb::ConstantSP>*> unwriteDdbVecs;
            std::vector<std::vector<py::object>*> unwritePyVecs;
            writer_->getPytoDdb()->getUnwrittenData(unwritePyVecs, unwriteDdbVecs);
            py::list pyunwrite(unwriteDdbVecs.size() + unwritePyVecs.size());
            int rowindex, colindex;
            rowindex = 0;
            for(auto &dbvec : unwriteDdbVecs){
                py::list pyrow(dbvec->size());
                colindex=0;
                for(auto &dvone : *dbvec){
                    pyrow[colindex++] = ddb::DdbPythonUtil::toPython(dvone);
                }
                pyunwrite[rowindex++] = pyrow;
                delete dbvec;
            }
            for(auto &pyvec : unwritePyVecs){
                py::list pyrow(pyvec->size());
                colindex = 0;
                for(auto &pyone : *pyvec){
                    pyrow[colindex++] =  pyone;
                }
                pyunwrite[rowindex++] = pyrow;
                delete pyvec;
            }
            return pyunwrite;
        } catch (ddb::RuntimeException &ex) {
            throw std::runtime_error(std::string("<Exception> in getUnwrittenData: ") + ex.what());
        }
    }
    py::dict insert(const py::args &args){
        if(writer_->isExit()){
            throw std::runtime_error(std::string("<Exception> in insert: thread is exiting."));
        }
        py::dict errorinfo;
        RECORD_TIME("insert.add");
        int size = args.size();
        if(size != writer_->getColSize()){
            errorinfo["errorCode"] = ddb::ErrorCodeInfo::formatApiCode(ddb::ErrorCodeInfo::EC_InvalidParameter);
            errorinfo["errorInfo"] = std::string("Column counts don't match ") + std::to_string(writer_->getColSize());
            return errorinfo;
        }
        std::vector<py::object> *pobjs=new std::vector<py::object>;
        pobjs->reserve(size);
        for (int i = 0; i < size; ++i){
            pobjs->push_back(py::reinterpret_borrow<py::object>(args[i]));
        }
        if(writer_->getPytoDdb()->add(pobjs) == false){
            delete pobjs;
            if(writer_->isExit()){
                throw std::runtime_error(std::string("<Exception> in insert: thread is exiting."));
            }
            errorinfo["errorCode"] = ddb::ErrorCodeInfo::formatApiCode(ddb::ErrorCodeInfo::EC_InvalidObject);
            errorinfo["errorInfo"] = std::string("Invalid object");
            return errorinfo;
        }
        errorinfo["errorCode"] = "";
        //ddb::g_OutputDestroyMsg=false;
        return errorinfo;
    }
    py::dict insertUnwrittenData(const py::list &records){
        if(writer_->isExit()){
            throw std::runtime_error(std::string("<Exception> in insert: thread is exiting."));
        }
        py::dict errorinfo;
        //std::vector<std::vector<ddb::ConstantSP>*> vectorOfVector(records.size());
        for(pybind11::size_t row = 0; row < records.size(); row++){
            py::list pylist = records[row];
            int size = pylist.size();
            if(size != writer_->getColSize()){
                errorinfo["errorCode"] = ddb::ErrorCodeInfo::formatApiCode(ddb::ErrorCodeInfo::EC_InvalidObject);
                errorinfo["errorInfo"] = std::string("arg size mismatch col size ") + std::to_string(writer_->getColSize());
                return errorinfo;
            }
            std::vector<py::object> *pobjs=new std::vector<py::object>;
            pobjs->reserve(size);
            for (int i = 0; i < size; ++i){
                pobjs->push_back(py::reinterpret_borrow<py::object>(pylist[i]));
            }
            if(writer_->getPytoDdb()->add(pobjs) == false){
                delete pobjs;
                if(writer_->isExit()){
                    throw std::runtime_error(std::string("<Exception> in insert: thread is exiting."));
                }
                errorinfo["errorCode"] = ddb::ErrorCodeInfo::formatApiCode(ddb::ErrorCodeInfo::EC_InvalidObject);
                errorinfo["errorInfo"] = std::string("Invalid object");
                return errorinfo;
            }
        }
        errorinfo["errorCode"] = "";
        return errorinfo;
    }
private:
    static std::unique_ptr<vector<string>> pylist2Stringvector(py::list pylist){
        std::unique_ptr<vector<string>> psites(new vector<string>);
        for (py::handle o : pylist) { psites->emplace_back(py::cast<std::string>(o)); }
        return psites;
    }
    static std::unique_ptr<vector<ddb::COMPRESS_METHOD>> pylist2Compressvector(py::list pylist){
        std::unique_ptr<vector<ddb::COMPRESS_METHOD>> pcompresstypes(new vector<ddb::COMPRESS_METHOD>);
        for (py::handle o : pylist) {
            std::string typeStr = py::cast<std::string>(o);
            transform(typeStr.begin(), typeStr.end(), typeStr.begin(), ::toupper);
            ddb::COMPRESS_METHOD type;
            if(typeStr == "LZ4"){
                type=ddb::COMPRESS_LZ4;
            }else if(typeStr == "DELTA"){
                type=ddb::COMPRESS_DELTA;
            }else{
                throw std::runtime_error(std::string("Unsupported compression method ") + typeStr);
            }
            pcompresstypes->emplace_back(type);
        }
        return pcompresstypes;
    }
    ddb::SmartPointer<ddb::MultithreadedTableWriter> writer_;
};

PYBIND11_MODULE(dolphindbcpp, m) {
    m.doc() = R"pbdoc(dolphindbcpp: this is a C++ boosted DolphinDB Python API)pbdoc";

    py::class_<DBConnectionPoolImpl>(m, "dbConnectionPoolImpl")
        .def(py::init<const std::string &,int,int,const std::string &,const std::string &,bool, bool, bool, bool, bool>())
        .def("run", (py::object(DBConnectionPoolImpl::*)(const std::string &, int)) & DBConnectionPoolImpl::run)
        .def("run", (py::object(DBConnectionPoolImpl::*)(const std::string &, int, const py::args &)) & DBConnectionPoolImpl::run)
        .def("run", (py::object(DBConnectionPoolImpl::*)(const std::string &, int, const py::kwargs &)) & DBConnectionPoolImpl::run)
        .def("run", (py::object(DBConnectionPoolImpl::*)(const std::string &, int, const py::args &, const py::kwargs &)) & DBConnectionPoolImpl::run)
        .def("isFinished",(bool(DBConnectionPoolImpl::*)(int)) & DBConnectionPoolImpl::isFinished)
        .def("getData",(py::object(DBConnectionPoolImpl::*)(int)) & DBConnectionPoolImpl::getData)
        .def("shutDown",&DBConnectionPoolImpl::shutDown)
        .def("getSessionId",&DBConnectionPoolImpl::getSessionId);

    py::class_<SessionImpl>(m, "sessionimpl")
        .def(py::init<bool,bool,int,bool,bool>())
        .def("connect", &SessionImpl::connect)
        .def("login", &SessionImpl::login)
        .def("getInitScript", &SessionImpl::getInitScript)
        .def("setInitScript", &SessionImpl::setInitScript)
        .def("close", &SessionImpl::close)
        .def("getSessionId", &SessionImpl::getSessionId)
        .def("run", (py::object(SessionImpl::*)(const std::string &)) & SessionImpl::run)
        .def("run", (py::object(SessionImpl::*)(const std::string &, const py::args &)) & SessionImpl::run)
        .def("run", (py::object(SessionImpl::*)(const std::string &, const py::kwargs &)) & SessionImpl::run)
        .def("run", (py::object(SessionImpl::*)(const std::string &, const py::args &, const py::kwargs &)) & SessionImpl::run)
        .def("runBlock",&SessionImpl::runBlock)
        .def("upload", &SessionImpl::upload)
        .def("nullValueToZero", &SessionImpl::nullValueToZero)
        .def("nullValueToNan", &SessionImpl::nullValueToNan)
        .def("enableStreaming", &SessionImpl::enableStreaming)
        .def("subscribe", &SessionImpl::subscribe)
        .def("subscribeBatch", &SessionImpl::subscribeBatch)
        .def("unsubscribe", &SessionImpl::unsubscribe)
        .def("hashBucket", &SessionImpl::hashBucket)
        .def("getSubscriptionTopics", &SessionImpl::getSubscriptionTopics)
        .def("loadPickleFile", &SessionImpl::loadPickleFile);
    
    py::class_<BlockReader>(m, "blockReader")
        .def(py::init<ddb::BlockReaderSP>())
        .def("read", (py::object(BlockReader::*)()) &BlockReader::read)
        .def("skipAll", &BlockReader::skipAll)
        .def("hasNext", (py::bool_(BlockReader::*)())&BlockReader::hasNext);

    py::class_<PartitionedTableAppender>(m, "partitionedTableAppender")
        .def(py::init<const std::string &,const std::string &,const std::string &,DBConnectionPoolImpl&>())
        .def("append", &PartitionedTableAppender::append);

    py::class_<AutoFitTableAppender>(m, "autoFitTableAppender")
        .def(py::init<const std::string &, const std::string&, SessionImpl&>())
        .def("append", &AutoFitTableAppender::append);

    py::class_<BatchTableWriter>(m, "batchTableWriter")
        .def(py::init<const std::string &,int,const std::string &,const std::string &,bool>())
        .def("addTable", &BatchTableWriter::addTable)
        .def("getStatus", &BatchTableWriter::getStatus)
        .def("getAllStatus", &BatchTableWriter::getAllStatus)
        .def("getUnwrittenData", &BatchTableWriter::getUnwrittenData)
        .def("removeTable", &BatchTableWriter::removeTable)
        .def("insert", &BatchTableWriter::insert);

    py::class_<MultithreadedTableWriter>(m, "multithreadedTableWriter")
        .def(py::init<const std::string &, int, const std::string &, const std::string &,const std::string &, const std::string &, bool, bool,const py::list &, int, float,int, const std::string&,const py::list &>())
        .def("getStatus", &MultithreadedTableWriter::getStatus)
        .def("getUnwrittenData", &MultithreadedTableWriter::getUnwrittenData)
        .def("insert", &MultithreadedTableWriter::insert)
        .def("insertUnwrittenData", &MultithreadedTableWriter::insertUnwrittenData)
        .def("waitForThreadCompletion", &MultithreadedTableWriter::waitForThreadCompletion);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
