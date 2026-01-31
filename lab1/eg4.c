#include<stdio.h>
#include<omp.h>
int main(){
    //removing race condition with reduction
    int i, sum = 0;
    #pragma omp paralle for reduction(+:sum)
    for(i = 1; i <= 100; i++){
        sum += i;
    }
    printf("Sum = %d\n", sum);
    return 0;
}