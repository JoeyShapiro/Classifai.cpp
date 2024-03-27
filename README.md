## Classifai.cpp
Started from bert.cpp and llama.cpp

## Setup

```bash
git submodule update --remote
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
cd ..
gcc main.cpp /build/libbert.dylib -Wl,-rpath,`pwd`/build/
```
