#include<stdio.h>
#include<omp.h>
#include<stdlib.h>

#define N 65536 //coz x and y are vectors of size 2^16

int main(){
    int a  = 2;
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double p1_start, p1_end, t_p1, pi_start, pi_end, t_pi;
    double speedup;

    //one thread
    p1_start = omp_get_wtime();
    for(int i = 0; i < N; i++){
        x[i] = a * x[i] + y[i];
    }
    p1_end = omp_get_wtime();
    t_p1 = p1_end - p1_start;
    speedup = t_p1 / t_p1;

    printf("1 Thread: Time = %f, SpeedUp = %f\n", t_p1, speedup);

    //n threads
    int maxThreads = omp_get_max_threads();
    for(int n = 2; n <= maxThreads; n++){
        omp_set_num_threads(n);
        pi_start = omp_get_wtime();
        #pragma omp parallel for
        for(int i = 0; i < N; i++){
            x[i] = a * x[i] + y[i];
        }
        pi_end = omp_get_wtime();
        t_pi = pi_end - pi_start;
        speedup = t_p1 / t_pi;

        printf("%d Thread: Time = %f, SpeedUp = %f\n", n, t_pi, speedup);
    }
    free(x);
    free(y);
    return 0;
}