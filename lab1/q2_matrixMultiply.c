#include <stdio.h>
#include <omp.h>

#define SIZE 1000
double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

int main() {
    for(int i=0; i<SIZE; i++) for(int j=0; j<SIZE; j++) { A[i][j]=1.0; B[i][j]=2.0; }
    double start, end, seq_time;

    start = omp_get_wtime();
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++) {
            double sum = 0;
            for (int k=0; k<SIZE; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    end = omp_get_wtime();
    seq_time = end - start;
    printf("Sequential Time: %f sec\n", seq_time);

    start = omp_get_wtime();
    #pragma omp parallel for num_threads(8)
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++) {
            double sum = 0;
            for (int k=0; k<SIZE; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    printf("1D Parallel Time: %f sec (Speedup: %fx)\n", omp_get_wtime()-start, seq_time/(omp_get_wtime()-start));

    start = omp_get_wtime();
    #pragma omp parallel for collapse(2) num_threads(8)
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++) {
            double sum = 0;
            for (int k=0; k<SIZE; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    printf("2D Parallel Time: %f sec (Speedup: %fx)\n", omp_get_wtime()-start, seq_time/(omp_get_wtime()-start));

    return 0;
}