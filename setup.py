import setuptools
import sys
import os
import re
from shutil import copyfile
from distutils.version import LooseVersion
import platform
import subprocess
from setuptools.dist import Distribution
from setuptools import Extension
from setuptools.command.build_ext import build_ext

def cvt(s):
    return s.replace('\\', '/').replace('C:/', '/cygdrive/c/')

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = []

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + cvt(extdir)]
            cmake_args += ['-DPYTHON_EXECUTABLE=' + cvt(sys.executable),
                           '-DPYTHON_INCLUDE=' + cvt(sys.exec_prefix) + '/include',
                           '-DPYTHON_LIBDIR=' + cvt(sys.exec_prefix) + '/libs',
                           '-DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc.exe',
                           '-DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++.exe'

            ]
            if sys.version_info[1] == 6:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.6']
            if sys.version_info[1] == 7:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.7']
            if sys.version_info[1] == 8:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.8']
            if sys.version_info[1] == 9:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.9']
            # if sys.maxsize > 2 ** 32:
                # cmake_args += ['-A', 'x64']
            build_args += ['--', '-j']
            ext.sourcedir = cvt(ext.sourcedir)
            self.build_temp = cvt(self.build_temp)
        else:
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                           '-DPYTHON_EXECUTABLE=' + sys.executable, 
                           '-DCMAKE_BUILD_TYPE=' + cfg]
            if sys.version_info[1] == 6:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.6']
            if sys.version_info[1] == 7:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.7']
            if sys.version_info[1] == 8:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.8']
            if sys.version_info[1] == 9:
                cmake_args += ['-DPICKLEAPI_LIBDIR=python3.9']
            build_args += ['--', '-j']
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


with open("README.md", "r") as f:
    long_description = f.read()

old_packages = setuptools.find_packages('src')
print("find_packages : ",len(old_packages))
packages = []
packages_dir = {'': 'src'}
isPy2 = sys.version_info[0] < 3

for pkg_name in old_packages:
    if 'dolphindb_numpy' in pkg_name:
        if isPy2:
            continue
        pkg_path = 'src/' + pkg_name.replace('.', '/')
        pkg_name = pkg_name.replace('dolphindb_numpy', 'numpy')
        packages_dir[pkg_name] = pkg_path
    packages.append(pkg_name)

setuptools.setup(name='dolphindb',
                 version='1.30.17.2',
                 install_requires=[
                     "future",
                     "numpy>=1.18,<=1.22.3",
                     "pandas>=0.25.1,!=1.3.0"
                 ],
                 author='DolphinDB, Inc.',
                 author_email='support@dolphindb.com',
                 url='https://www.dolphindb.com',
                 license='DolphinDB',
                 description='A C++ boosted DolphinDB API based on Pybind11',
                 long_description=long_description,
                 packages=packages,
                 package_dir=packages_dir,
                 ext_modules=[CMakeExtension('dolphindb/dolphindbcpp')],
                 cmdclass=dict(build_ext=CMakeBuild),
                 test_suite='tests',
                 platforms=[
                     "Windows",
                     "Linux",
                 ],
                 classifiers=[
                     "Operating System :: POSIX :: Linux",
                     "Operating System :: Microsoft :: Windows",
                     "Programming Language :: Python :: 2",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3",
                     "Programming Language :: Python :: 3.4",
                     "Programming Language :: Python :: 3.5",
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: Implementation :: CPython",
                 ],
                 python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*",
                 distclass=BinaryDistribution)
