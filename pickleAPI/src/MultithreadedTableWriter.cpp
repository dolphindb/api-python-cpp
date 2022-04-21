#include "MultithreadedTableWriter.h"
#include "ScalarImp.h"
#include <thread>
#include "DdbPythonUtil.h"

namespace dolphindb{

#define DLOG //DLogger::Info

MultithreadedTableWriter::MultithreadedTableWriter(const std::string& hostName, int port, const std::string& userId, const std::string& password,
										const string& dbName, const string& tableName, bool useSSL,
										bool enableHighAvailability, const vector<string> *pHighAvailabilitySites,
										int batchSize, float throttle, int threadCount, const string& partitionCol,
										const vector<COMPRESS_METHOD> *pCompressMethods):
                        dbName_(dbName),
                        tableName_(tableName),
                        batchSize_(batchSize),
                        throttleMilsecond_(throttle*1000),
						hasError_(false),
                        pytoDdb_(new PytoDdbRowPool(*this))
                        {
    if(threadCount < 1){
        throw RuntimeException("The parameter threadCount must be greater than or equal to 1");
    }
    if(batchSize < 1){
        throw RuntimeException("The parameter batchSize must be greater than or equal to 1");
    }
    if(throttle < 0){
        throw RuntimeException("The parameter throttle must be greater than 0");
    }
	if (threadCount > 1 && partitionCol.empty()) {
		throw RuntimeException("The parameter partitionCol must be specified when threadCount is greater than 1");
	}
	bool isCompress = false;
	int keepAliveTime = 7200;
	if (pCompressMethods != NULL && pCompressMethods->size() > 0) {
		for (auto one : *pCompressMethods) {
			if (one != COMPRESS_DELTA && one != COMPRESS_LZ4) {
				throw RuntimeException("Unsupported compression method "+one);
			}
		}
		compressMethods_ = *pCompressMethods;
		isCompress = true;
	}
    SmartPointer<DBConnection> pConn=new DBConnection(useSSL, false, keepAliveTime, isCompress);
	vector<string> highAvailabilitySites;
	if (pHighAvailabilitySites != NULL) {
		highAvailabilitySites.assign(pHighAvailabilitySites->begin(), pHighAvailabilitySites->end());
	}
    bool ret = pConn->connect(hostName, port, userId, password, "", enableHighAvailability, highAvailabilitySites);
    if(!ret){
        throw RuntimeException("Failed to connect to server "+hostName+":"+std::to_string(port));
    }

    DictionarySP schema;
    if(tableName.empty()){
        schema = pConn->run("schema(" + dbName + ")");
    }else{
        schema = pConn->run(std::string("schema(loadTable(\"") + dbName + "\",\"" + tableName + "\"))");
    }
    ConstantSP partColNames = schema->getMember("partitionColumnName");
    if(partColNames->isNull()==false){//partitioned table
        isPartionedTable_ = true;
    }else{//Not partitioned table
        if (tableName.empty() == false) {//Single partitioned table
			if (threadCount > 1) {
				throw RuntimeException("The parameter threadCount must be 1 for a dimension table");
			}
		}
		isPartionedTable_ = false;
    }

    TableSP colDefs = schema->getMember("colDefs");

    ConstantSP colDefsTypeInt = colDefs->getColumn("typeInt");
    size_t columnSize = colDefs->size();
	if (compressMethods_.size() > 0 && compressMethods_.size() != columnSize) {
		throw RuntimeException("The number of elements in parameter compressMethods does not match the column size "+std::to_string(columnSize));
	}
    
    ConstantSP colDefsName = colDefs->getColumn("name");
    ConstantSP colDefsTypeString = colDefs->getColumn("typeString");
    for(size_t i = 0; i < columnSize; i++){
        colNames_.push_back(colDefsName->getString(i));
        colTypes_.push_back(static_cast<DATA_TYPE>(colDefsTypeInt->getInt(i)));
        colTypeString_.push_back(colDefsTypeString->getString(i));
    }
	if (threadCount > 1) {//Only multithread need partition col info
		if (isPartionedTable_) {
			ConstantSP partitionSchema;
			int partitionType;
			DATA_TYPE partitionColType;
			if (partColNames->isScalar()) {
				if (partColNames->getString() != partitionCol) {
					throw RuntimeException("The parameter partionCol must be the partitioning column '" + partColNames->getString() + "' in the partitioned table");
				}
				partitionColumnIdx_ = schema->getMember("partitionColumnIndex")->getInt();
				partitionSchema = schema->getMember("partitionSchema");
				partitionType = schema->getMember("partitionType")->getInt();
				partitionColType = (DATA_TYPE)schema->getMember("partitionColumnType")->getInt();
			}
			else {
				int dims = partColNames->size();
				if (dims > 1 && partitionCol.empty()) {
					throw RuntimeException("The parameter partitionCol must be specified for a partitioned table");
				}
				int index = -1;
				for (int i = 0; i < dims; ++i) {
					if (partColNames->getString(i) == partitionCol) {
						index = i;
						break;
					}
				}
				if (index < 0)
					throw RuntimeException("The parameter partionCol must be the partitioning columns in the partitioned table");
				partitionColumnIdx_ = schema->getMember("partitionColumnIndex")->getInt(index);
				partitionSchema = schema->getMember("partitionSchema")->get(index);
				partitionType = schema->getMember("partitionType")->getInt(index);
				partitionColType = (DATA_TYPE)schema->getMember("partitionColumnType")->getInt(index);
			}
			if (colTypes_[partitionColumnIdx_] >= ARRAY_TYPE_BASE) {//arrayVector can't be partitioned
				throw RuntimeException("The parameter partitionCol cannot be array vector");
			}
			partitionDomain_ = Util::createDomain((PARTITION_TYPE)partitionType, partitionColType, partitionSchema);
		}
		else {//isPartionedTable_==false
			if (partitionCol.empty() == false) {
				int threadcolindex = -1;
				for (unsigned int i = 0; i < colNames_.size(); i++) {
					if (colNames_[i] == partitionCol) {
						threadcolindex = i;
						break;
					}
				}
				if (threadcolindex < 0) {
					throw RuntimeException("No match found for " + partitionCol);
				}
				if (colTypes_[threadcolindex] >= ARRAY_TYPE_BASE) {//arrayVector can't be partitioned
					throw RuntimeException("The parameter partitionCol cannot be array vector");
				}
				threadByColIndexForNonPartion_ = threadcolindex;
			}
		}
	}
    {
        if(tableName_.empty()){//memory table
            scriptTableInsert_ = std::move(std::string("tableInsert{\"") + dbName_ + "\"}");
        }else if(isPartionedTable_){//partitioned table
            scriptTableInsert_ = std::move(std::string("tableInsert{loadTable(\"") + dbName_ + "\",\"" + tableName_ + "\")}");
        }else{//single partitioned table
            scriptTableInsert_ = std::move(std::string("tableInsert{loadTable(\"") + dbName_ + "\",\"" + tableName_ + "\")}");
            //Remove support for disk table
            /*{
                std::string tempTableName = "tmp" +  tableWriter_.tableName_;
                std::string colNames;
                std::string colTypes;
                for(unsigned int i = 0; i < tableWriter_.colNames_.size(); i++){
                    colNames += "`" + tableWriter_.colNames_[i];
                    colTypes += "`" + tableWriter_.colTypeString_[i];
                }
                std::string scriptCreateTmpTable = std::move(std::string("tempTable = table(") + "1000:0," + colNames + "," + colTypes + ")");
                try{
                    writeThread_.conn->run(scriptCreateTmpTable);
                }catch(std::exception &e){
                    DLogger::Error("threadid=", writeThread_.threadId, " Init table error: ", e.what()," script:", scriptCreateTmpTable);
                    tableWriter_.setDestroyed(ErrorCodeInfo::EC_Server,std::string("Init table error: ")+e.what()+" script: "+scriptCreateTmpTable);
                    //std::cerr << Util::createTimestamp(Util::getEpochTime())->getString() << " Backgroud thread of table (" << tableWriter_.dbName_ << " " << tableWriter_.tableName_ << "). Failed to init data to server, with exception: " << e.what() << std::endl;
                    return false;
                }
            }
            writeThread_.scriptTableInsert = std::move(std::string("tableInsert{tempTable}"));
            writeThread_.scriptSaveTable = std::move(std::string("saveTable(database(\"") + tableWriter_.dbName_ + "\")" + ",tempTable,\"" + tableWriter_.tableName_ + "\", 1);tempTable.clear!();");
            */
        }
    }
    // init done, start thread now.
    threads_.resize(threadCount);
    for(unsigned int i = 0; i < threads_.size(); i++){
        WriterThread &writerThread = threads_[i];
        writerThread.threadId = 0;
        writerThread.sentRows = 0;
        writerThread.sendingRows = 0;
		writerThread.exit = false;
        writerThread.idleSem.release();
        if(i==0){
            writerThread.conn=pConn;
        }else{
            writerThread.conn = new DBConnection(useSSL, false, keepAliveTime, isCompress);
            if(writerThread.conn->connect(hostName, port, userId, password, "", enableHighAvailability, highAvailabilitySites)==false){
                throw RuntimeException("Failed to connect to server "+hostName+":"+std::to_string(port));
            }
        }
        writerThread.writeThread = new Thread(new SendExecutor(*this,writerThread));
        writerThread.writeThread->start();
    }
	insertExecutor_ = new InsertExecutor(*this);
	insertThread_ = new Thread(insertExecutor_);
	insertThread_->start();
}

MultithreadedTableWriter::~MultithreadedTableWriter(){
    waitForThreadCompletion();
    {
        std::vector<ConstantSP>* pitem = NULL;
        for(auto &thread : threads_){
            while(thread.writeQueue.pop(pitem)){
                delete pitem;
            }
            while(thread.failedQueue.pop(pitem)){
                delete pitem;
            }
        }
    }
}

void MultithreadedTableWriter::waitForThreadCompletion() {
    if(pytoDdb_->isExit())
        return;
    pytoDdb_->startExit();
    insertExecutor_->exit();
	insertThread_->join();
    for(auto &thread : threads_){
		thread.exit = true;
		thread.nonemptyNotify.notify();
    }
	for(auto &thread : threads_){
		thread.writeThread->join();
    }
    for(auto &thread : threads_){
        thread.conn->close();
    }
    pytoDdb_->endExit();
	setError(0, "");
    //DLogger::Info(RecordTime::printAllTime());
}

void MultithreadedTableWriter::setError(int code, const string &info){
    LockGuard<Mutex> LockGuard(&tableMutex_);
    if(hasError_)
        return;
    errorInfo_.set(code, info);
	hasError_ = true;
}

bool MultithreadedTableWriter::SendExecutor::init(){
	writeThread_.threadId = Util::getCurThreadId();
    return true;
}

void MultithreadedTableWriter::insert(std::vector<ConstantSP> **records, int recordCount){
    RECORDTIME("MTW:insert");
	//To speed up, ignore check
	/*
	for (auto &row : vectorOfVector) {
		if (row.size() != colNames_.size()) {
			errorInfo.set(ErrorCodeInfo::EC_InvalidObject, "Invalid vector size " + std::to_string(row.size()) + ", expect " + std::to_string(colNames_.size()));
			return false;
		}
		int index = 0;
		DATA_TYPE dataType;
		for (auto &param : row) {
			dataType = getColDataType(index);
			if (param->getType() != dataType && dataType != DATA_TYPE::DT_SYMBOL) {
				errorInfo.set(ErrorCodeInfo::EC_InvalidObject, "Object type mismatch " + Util::getDataTypeString(param->getType()) + ", expect " + Util::getDataTypeString(dataType));
				return false;
			}
			index++;
		}
	}
	*/
    if(threads_.size() > 1){
        if(isPartionedTable_){
			VectorSP pvector = Util::createVector(getColDataType(partitionColumnIdx_), 0, recordCount);
            for(int i=0; i < recordCount; i++){
                pvector->append(records[i]->at(partitionColumnIdx_));
            }
            vector<int> threadindexes = partitionDomain_->getPartitionKeys(pvector);
            for(unsigned int row = 0; row < threadindexes.size(); row++){
                insertThreadWrite(threadindexes[row], records[row]);
            }
        }else{
            int threadindex;
            for(int i=0; i < recordCount; i++){
                threadindex = records[i]->at(threadByColIndexForNonPartion_)->getHash(threads_.size());
                insertThreadWrite(threadindex, records[i]);
            }
        }
    }else{
        for(int i=0; i < recordCount; i++){
            insertThreadWrite(0, records[i]);
        }
    }
}

void MultithreadedTableWriter::getStatus(Status &status){
    status.isExiting = hasError_;
    status.errorInfo = errorInfo_;
	status.sentRows = status.unsentRows = status.sendFailedRows = 0;
	status.threadStatus.resize(threads_.size() + 1);
    {
        ThreadStatus &threadStatus = status.threadStatus[0];
        insertExecutor_->getStatus(threadStatus);
        status.plus(threadStatus);
    }
	for(auto i = 0; i < threads_.size(); i++){
        ThreadStatus &threadStatus = status.threadStatus[i + 1];
        WriterThread &writeThread = threads_[i];
        LockGuard<Mutex> LockGuard(&writeThread.writeMutex);
        threadStatus.threadId = writeThread.threadId;
        threadStatus.sentRows = writeThread.sentRows;
        threadStatus.unsentRows = writeThread.writeQueue.size() + writeThread.sendingRows;
        threadStatus.sendFailedRows = writeThread.failedQueue.size();
        status.plus(threadStatus);
    }
}

void MultithreadedTableWriter::getUnwrittenData(std::vector<std::vector<ConstantSP>*> &unwrittenData){
    insertExecutor_->getUnwrittenData(unwrittenData);
    for(auto &writeThread : threads_){
        SemLock idleLock(writeThread.idleSem);
        idleLock.acquire();
        writeThread.failedQueue.pop(unwrittenData, writeThread.failedQueue.size());
        writeThread.writeQueue.pop(unwrittenData, writeThread.writeQueue.size());
    }
}

void MultithreadedTableWriter::insertThreadWrite(int threadhashkey, std::vector<ConstantSP> *prow){
    if(threadhashkey < 0){
        threadhashkey = 0;
    }
    int threadIndex = threadhashkey % threads_.size();
    WriterThread &writerThread = threads_[threadIndex];
    writerThread.writeQueue.push(prow);
    writerThread.nonemptyNotify.notify();
}

void MultithreadedTableWriter::SendExecutor::run(){
    if(init()==false){
        return;
    }
	long batchWaitTimeout = 0, diff;
    while(isExit() == false){
        {
            RECORDTIME("MTW:wait");
            if(writeThread_.writeQueue.size() < 1){//Wait for first data
				writeThread_.nonemptyNotify.wait();
            }
            if (isExit())
                break;
            //wait for batchsize
            if (tableWriter_.batchSize_ > 1 && tableWriter_.throttleMilsecond_ > 0) {
                batchWaitTimeout = Util::getEpochTime() + tableWriter_.throttleMilsecond_;
                while (isExit() == false && writeThread_.writeQueue.size() < tableWriter_.batchSize_) {//check batchsize
                    diff = batchWaitTimeout - Util::getEpochTime();
                    if (diff > 0) {
                        writeThread_.nonemptyNotify.wait(diff);
                    }
                    else {
                        break;
                    }
                }
            }
        }
        while (isExit() == false && writeAllData());//write all data
    }
    //write left data
    while (tableWriter_.hasError_ == false && writeAllData());
}

bool MultithreadedTableWriter::SendExecutor::writeAllData(){
    //reset idle
    DLOG("writeAllData",writeThread_.writeQueue.size());
    SemLock idleLock(writeThread_.idleSem);
    idleLock.acquire();
    std::vector<std::vector<ConstantSP>*> items;
    {
        long size = writeThread_.writeQueue.size();
        if (size < 1){
            return false;
        }
        if(size > 65535)
            size = 65535;
        items.reserve(size);
        writeThread_.writeQueue.pop(items, size);
    }
    int size = items.size();
    if(size < 1){
        return false;
    }
    DLOG("writeAllData",size,"/",writeThread_.writeQueue.size());
    writeThread_.sendingRows = size;
    string runscript;
	bool writeOK = true;
    try{
        TableSP writeTable;
		int addRowCount = 0;
        {//create table
            RECORDTIME("MTW:createTable");
            writeTable = Util::createTable(tableWriter_.colNames_, tableWriter_.colTypes_, 0, size);
			writeTable->setColumnCompressMethods(tableWriter_.compressMethods_);
            INDEX insertedRows;
            std::string errMsg;
            for (int i = 0; i < size; i++){
                if(writeTable->append(*items[i], insertedRows, errMsg) == false){
                    tableWriter_.setError(ErrorCodeInfo::EC_InvalidObject, "Failed to append data to the table: "+errMsg);
					writeOK = false;
                    break;
                }
				addRowCount++;
            }
        }
		if(writeOK && addRowCount > 0){//save table
            RECORDTIME("MTW:saveTable");
            std::vector<ConstantSP> args;
            args.reserve(1);
            args.push_back(writeTable);
            runscript = tableWriter_.scriptTableInsert_;
            ConstantSP constsp = writeThread_.conn->run(runscript, args);
			int addresult = constsp->getInt();
			if (addresult != addRowCount) {
				std::cout << "Write complete size " << addresult << " mismatch insert size "<< addRowCount;
			}
            if (tableWriter_.scriptSaveTable_.empty() == false){
                runscript = tableWriter_.scriptSaveTable_;
                writeThread_.conn->run(runscript);
            }
            {
                LockGuard<Mutex> LockGuard(&writeThread_.writeMutex);
                writeThread_.sentRows += addRowCount;
                writeThread_.sendingRows = 0;
            }
            for(auto &one : items){
                delete one;
            }
        }
    }catch (std::exception &e){
        DLogger::Error("threadid=", writeThread_.threadId, " Failed to save the inserted data: ", e.what()," script:", runscript);
        tableWriter_.setError(ErrorCodeInfo::EC_Server,std::string("Failed to save the inserted data: ")+e.what()+" script: "+runscript);
		writeOK = false;
    }
    if (writeOK == false){
        LockGuard<Mutex> LockGuard(&writeThread_.writeMutex);
        for (auto &unwriteItem : items)
            writeThread_.failedQueue.push(unwriteItem);
        writeThread_.sendingRows = 0;
    }
    return true;
}

MultithreadedTableWriter::InsertExecutor::InsertExecutor(MultithreadedTableWriter &tableWriter) :
		tableWriter_(tableWriter),
        exitWhenEmpty_(false),
        insertingCount_(0){
    idle_.release();
}

void MultithreadedTableWriter::InsertExecutor::getUnwrittenData(std::vector<std::vector<ConstantSP>*> &unwrittenData){
    SemLock idleLock(idle_);
    idleLock.acquire();
    LockGuard<Mutex> LockGuard(&mutex_);
    while(!rows_.empty()){
        unwrittenData.push_back(rows_.front());
        rows_.pop();
    }
    nonempty_.reset();
}

MultithreadedTableWriter::InsertExecutor::~InsertExecutor() {
    LockGuard<Mutex> LockGuard(&mutex_);
	while(!rows_.empty()){
		delete rows_.front();
        rows_.pop();
    }
}

void MultithreadedTableWriter::InsertExecutor::getStatus(MultithreadedTableWriter::ThreadStatus &threadStatus){
    LockGuard<Mutex> LockGuard(&mutex_);
    threadStatus.unsentRows += rows_.size() + insertingCount_;
}

void MultithreadedTableWriter::InsertExecutor::run() {
	vector<std::vector<ConstantSP>*> insertRows;
	while (tableWriter_.hasError_ == false) {
        nonempty_.wait();
        SemLock idleLock(idle_);
        idleLock.acquire();
        {
            RECORDTIME("MTW:InsertExecutor");
            LockGuard<Mutex> LockGuard(&mutex_);
            int size = rows_.size();
            if(size < 1){
                if(exitWhenEmpty_)
                    break;
                nonempty_.reset();
                continue;
            }
			if (size > 65535)
				size = 65535;
			insertRows.reserve(size);
            while(!rows_.empty()){
                insertRows.push_back(rows_.front());
                rows_.pop();
                insertingCount_++;
                if(insertRows.size() >= size)
                    break;
            }
		}
        if(!insertRows.empty()){
            tableWriter_.insert(insertRows);
            insertRows.clear();
            insertingCount_ = 0;
        }
	}
}

};
