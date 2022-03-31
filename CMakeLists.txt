cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 1) 
project(api-python-cpp)
add_subdirectory(contrib/pybind11)


#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dolphindb)

# set compile flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
if (WIN32)
    include_directories(pickleAPI/include)
   # target_include_directories(test PRIVATE "c:/openssl-1.0.2j/static/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWINDOWS -D_hypot=hypot -DMS_WIN64")
	set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/mingw_x64/${PICKLEAPI_LIBDIR})
    set(DDB_LIB ${LIB_DIR}/libDolphinDBAPI.dll)
    set(SSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/mingw_x64/libssl.a)
    set(CRYPTO_LIB ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/mingw_x64/libcrypto.a)
else ()
    include_directories(pickleAPI/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX -D_GLIBCXX_USE_CXX11_ABI=0")
    set(DDB_LIB ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/linux_x64/${PICKLEAPI_LIBDIR}/libDolphinDBAPI.a)
    set(SSL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/linux_x64/libssl.a)
    set(CRYPTO_LIB ${CMAKE_CURRENT_SOURCE_DIR}/pickleAPI/bin/linux_x64/libcrypto.a)

endif ()

set(SOURCE_DIR "src/dolphindb/")
include_directories(${SOURCE_DIR})
include_directories(${PYTHON_INCLUDE})
link_directories(${PYTHON_LIBDIR})
message(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
message("PYTHON_INCLUDE: ${PYTHON_INCLUDE}")
message("PYTHON_LIBDIR: ${PYTHON_LIBDIR}")
message("PICKLEAPI_LIBDIR: ${PICKLEAPI_LIBDIR}")
set(SOURCES "${SOURCE_DIR}/binding.cpp")

link_directories(pickleAPI/bin/linux_x64/${PICKLEAPI_LIBDIR})
pybind11_add_module(dolphindbcpp ${SOURCES})

#add_library(source-combined STATIC "")
#set_target_properties(source-combined PROPERTIES LINKER_LANGUAGE CXX)

#target_link_libraries(source-combined PUBLIC
#  -Wl,--whole-archive
#  ${SSL_LIB}
#  ${CRYPTO_LIB}
#  -Wl,--no-whole-archive
#)

if(WIN32)
target_link_libraries(dolphindbcpp PUBLIC
        ${DDB_LIB}
        ${SSL_LIB}
        ${CRYPTO_LIB}
        uuid
        )
else()
target_link_libraries(dolphindbcpp PUBLIC
        ${DDB_LIB}
        ${SSL_LIB}
        ${CRYPTO_LIB}
        uuid
        rt
        )
endif()
		
if (WIN32)
set(WIN32_LIBS 
	libDolphinDBAPI.dll 
	libgcc_s_seh-1.dll 
	libstdc++-6.dll
	ssleay32.dll
	libeay32.dll
	)

foreach(file_i ${WIN32_LIBS})
	add_custom_command(
		TARGET dolphindbcpp
		POST_BUILD
		COMMAND "${CMAKE_COMMAND}" -E copy
		${LIB_DIR}/${file_i} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	)
endforeach(file_i)
endif(WIN32)
