#!/bin/bash

for ver in 36 37 38 39; do
    py=/cygdrive/c/Python$ver/python
    $py -m pip install wheel
    $py setup.py bdist_wheel -d wheelhouse
done
