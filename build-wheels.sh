#!/bin/bash
set -e -x

# Install a system package required by our library
#yum install -y atlas-devel
#yum install -y cmake
#yum install -y gcc
#yum install -y uuid

# Compile wheels
cd /io
#for PYBIN in /opt/python/*/bin; do
#    if [ "${PYBIN}" = "/opt/python/cp36-cp36m/bin" ] || [ "${PYBIN}" = "/opt/python/cp37-cp37m/bin" ] || [ "${PYBIN}" = "/opt/python/cp38-cp38/bin" ] || [ "${PYBIN}" = "/opt/python/cp39-cp39/bin" ]
#    then 
#        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ${PYBIN}"
#        #"${PYBIN}/pip" install -r /io/dev-requirements.txt
#        #"${PYBIN}/pip" wheel /io/ -w wheelhouse/
#        "${PYBIN}/python" setup.py bdist_wheel
#    fi
#done

/opt/python/cp38-cp38/bin/python setup.py bdist_wheel

mkdir -p wheelhouse

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ${PYBIN}"
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Install packages and test
#for PYBIN in /opt/python/*/bin/; do
#    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
#    "${PYBIN}/pip" install dolphindb --no-index -f /io/wheelhouse
#done
