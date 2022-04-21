#ifndef MUTITHREADEDTABLEWRITER_H_
#define MUTITHREADEDTABLEWRITER_H_

#include "Concurrent.h"
#include "DolphinDB.h"
#include "Util.h"
#include "Types.h"
#include "Exceptions.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <tuple>
#include <cassert>
#include <unordered_map>

#ifdef _MSC_VER
	#define EXPORT_DECL _declspec(dllexport)
#else
	#define EXPORT_DECL 
#endif

#ifndef RECORDTIME
#undef RECORDTIME
#endif
#define RECORDTIME //RecordTime _recordTime

namespace dolphindb{

class PytoDdbRowPool;
class EXPORT_DECL  MultithreadedTableWriter {
public:
    struct ThreadStatus{
        long threadId;
        long sentRows,unsentRows,sendFailedRows;
        ThreadStatus(){
            threadId = 0;
            sentRows = unsentRows = sendFailedRows = 0;
        }
    };
    struct Status{
        bool isExiting;
        ErrorCodeInfo errorInfo;
		long sentRows, unsentRows, sendFailedRows;
        std::vector<ThreadStatus> threadStatus;
        void plus(const ThreadStatus &threadStatus){
            sentRows += threadStatus.sentRows;
            unsentRows += threadStatus.unsentRows;
            sendFailedRows += threadStatus.sendFailedRows;
        }
    };
    /**
     * If fail to connect to the specified DolphinDB server, this function throw an exception.
     */
    MultithreadedTableWriter(const std::string& host, int port, const std::string& userId, const std::string& password,
                            const string& dbPath, const string& tableName, bool useSSL, bool enableHighAvailability = false, const vector<string> *pHighAvailabilitySites = NULL,
							int batchSize = 1, float throttle = 0.01f,int threadCount = 1, const string& partitionCol ="",
							const vector<COMPRESS_METHOD> *pCompressMethods = NULL);

    ~MultithreadedTableWriter();

    void getStatus(Status &status);
    void getUnwrittenData(std::vector<std::vector<ConstantSP>*> &unwrittenData);
    void insert(std::vector<ConstantSP> **records, int recordCount);
	void insert(std::vector<std::vector<ConstantSP>*> &records) { insert(records.data(), records.size()); }
    void waitForThreadCompletion();
    bool isExit(){ return hasError_; }

    const DATA_TYPE* getColType(){ return colTypes_.data(); }
    int getColSize(){ return colTypes_.size(); }

    template<typename... TArgs>
    bool insert(ErrorCodeInfo &errorInfo, TArgs... args){
        RECORDTIME("MTW:insertValue");
        if(hasError_){
			throw RuntimeException("Thread is exiting.");
        }
		{
			auto argSize = sizeof...(args);
			if (argSize != colTypes_.size()) {
				errorInfo.set(ErrorCodeInfo::EC_InvalidParameter, "Column counts don't match "+std::to_string(argSize));
				return false;
			}
		}
		{
			errorInfo.clearError();
			int colIndex = 0;
			ConstantSP result[] = { Util::createObject(getColDataType(colIndex++), args, &errorInfo)... };
			if (errorInfo.hasError())
				return false;
			insertExecutor_->add(result, colTypes_.size());
		}
        return true;
    }
    void setError(int code, const string &info);

private:
    DATA_TYPE getColDataType(int colIndex) {
		DATA_TYPE dataType = colTypes_[colIndex];
		if (dataType >= ARRAY_TYPE_BASE)
			dataType = (DATA_TYPE)(dataType - ARRAY_TYPE_BASE);
		return dataType;
	}
	void insertThreadWrite(int threadhashkey, std::vector<ConstantSP> *prow);

    struct WriterThread{
        SmartPointer<DBConnection> conn;
        
        SynchronizedQueue<std::vector<ConstantSP>*> writeQueue;
        SynchronizedQueue<std::vector<ConstantSP>*> failedQueue;
        ThreadSP writeThread;
        ConditionalNotifier nonemptyNotify;

        Mutex writeMutex;
        Semaphore idleSem;
        unsigned int threadId;
        long sentRows, sendingRows;
		bool exit;
    };
    class SendExecutor : public dolphindb::Runnable {
    public:
		SendExecutor(MultithreadedTableWriter &tableWriter,WriterThread &writeThread):
                        tableWriter_(tableWriter),
                        writeThread_(writeThread){};
        virtual void run();
    private:
		bool isExit() { return tableWriter_.hasError_ || writeThread_.exit; }
        bool init();
        bool writeAllData();
        MultithreadedTableWriter &tableWriter_;
        WriterThread &writeThread_;
    };
	class InsertExecutor : public dolphindb::Runnable {
	public:
		InsertExecutor(MultithreadedTableWriter &tableWriter);
		~InsertExecutor();
		virtual void run();
		void exit() {
            exitWhenEmpty_ = true;
            nonempty_.set();
		}
		void add(const ConstantSP* row, int size){
			std::vector<ConstantSP> *pvectorRow = new std::vector<ConstantSP>;
			pvectorRow->resize(size);
			for (int i = 0; i < size; i++)
				pvectorRow->at(i) = row[i];
            LockGuard<Mutex> LockGuard(&mutex_);
			rows_.push(pvectorRow);
            nonempty_.set();
		}
        void add(const vector<vector<ConstantSP>*> &rows){
            LockGuard<Mutex> LockGuard(&mutex_);
            for(auto &one : rows)
			    rows_.push(one);
            nonempty_.set();
		}
        void getUnwrittenData(std::vector<std::vector<ConstantSP>*> &unwrittenData);
        void getStatus(MultithreadedTableWriter::ThreadStatus &threadStatus);
	private:
		MultithreadedTableWriter &tableWriter_;
        int insertingCount_;
        bool exitWhenEmpty_;
        Semaphore idle_;
        Signal nonempty_;
        Mutex mutex_;
		std::queue<vector<ConstantSP>*> rows_;
	};
    
private:
    friend class SendExecutor;
	friend class InsertExecutor;
    const std::string dbName_;
    const std::string tableName_;
    const int batchSize_;
    const int throttleMilsecond_;
    bool isPartionedTable_, hasError_;
    std::vector<string> colNames_,colTypeString_;
    std::vector<DATA_TYPE> colTypes_;
	std::vector<COMPRESS_METHOD> compressMethods_;
	//Following parameters only valid in multithread mode
    SmartPointer<Domain> partitionDomain_;
    int partitionColumnIdx_;
    int threadByColIndexForNonPartion_;
	//End of following parameters only valid in multithread mode
    std::vector<WriterThread> threads_;
	ThreadSP insertThread_;
	SmartPointer<InsertExecutor> insertExecutor_;
    Mutex tableMutex_;
    ErrorCodeInfo errorInfo_;
    std::string scriptTableInsert_;
    std::string scriptSaveTable_;
    friend class PytoDdbRowPool;
    PytoDdbRowPool *pytoDdb_;
public:
    PytoDdbRowPool * getPytoDdb(){ return pytoDdb_;}
};

};

#endif //MUTITHREADEDTABLEWRITER_H_