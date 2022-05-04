#ifndef __UTILS__
#define __UTILS__
#include <cstdlib>
#include <cstdio>
template <typename T>
void rand_array(T* array,size_t len){
    for(int i=0;i<len;++i){
        array[i]=((float))rand()/RAND_MAX;
    }
}
#endif