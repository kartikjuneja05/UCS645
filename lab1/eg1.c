#include<stdio.h>
#include<omp.h>
int main(){
    #pragma omp parallel
    {
        printf("Hello from thread no %d\n", omp_get_thread_num());
    }
    return 0;
}