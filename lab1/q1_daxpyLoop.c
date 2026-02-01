#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 65536 

int main() {
    double a = 2.0;
    double *x = (double*)malloc(N * sizeof(double));
    double *y = (double*)malloc(N * sizeof(double));
    double start, end, seq_time, par_time;

    for(int i=0; i<N; i++) { x[i] = i*1.0; y[i] = i*2.0; }

    start = omp_get_wtime();
    for(int i=0; i<N; i++) {
        x[i] = a * x[i] + y[i];
    }
    end = omp_get_wtime();
    seq_time = end - start;
    printf("Sequential Time: %f sec\n", seq_time);

    printf("\nThreads\tTime (sec)\tSpeedup\n");
    for (int t = 2; t <= 16; t++) {
        start = omp_get_wtime();
        #pragma omp parallel for num_threads(t)
        for(int i=0; i<N; i++) {
            x[i] = a * x[i] + y[i];
        }
        end = omp_get_wtime();
        par_time = end - start;
        printf("%d\t%f\t%f\n", t, par_time, seq_time / par_time);
    }

    free(x); free(y);
    return 0;
}