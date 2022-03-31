## How to build

### Change the version number

Change it in file `setup.py`, one version number can only be used once for each upload to pypi.

### Build Linux Wheel

You can build linux wheels on linux or windows, using docker

1. on Linux

First build the docker image, then use build wheels with docker

```bash
docker build -t dev-ssl .
./build.sh
```

Output wheels will be located at `wheelhouse` directory.

2. on windows

Almost the same as linux, be careful with line ending of the file `build-wheels.sh`, if something is wrong, use `dos2linux`
to fix it.

with Windows Command Line (cmd), 
```cmd
docker build -t dev.
docker run --rm -e PLAT=manylinux2010_x86_64 -v %cd%:/io dev /io/build-wheels.sh
```

with PowerShell, use `${PWD}`
```powershell
docker build -t dev.
docker run --rm -e PLAT=manylinux2010_x86_64 -v ${PWD}:/io dev /io/build-wheels.sh
```

todo: may be write a build.bat

### Build Windows Wheels

You can build this only in windows for now.

1. Install `python27`, `python34`, `python35`, `python36`, `python37`, in `C:\`
2. Install cygwin, with `mingw-w64`
3. In cygwin, run `build-win.sh`

Output wheels will be located at `wheelhouse` directory too.

## Upload to pypi.org

```cmd
$ C:Python37\python.exe -m twine upload -u dolphindbpy -p DBpy@3817 .\wheelhouse\*`
```
