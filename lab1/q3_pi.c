#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;

int main() {
    double x, pi, sum = 0.0;
    double step = 1.0 / (double)num_steps;
    double start, end, seq_time;

    start = omp_get_wtime();
    for (int i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    end = omp_get_wtime();
    seq_time = end - start;
    printf("Sequential Pi: %.10f, Time: %f sec\n", pi, seq_time);

    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) private(x) num_threads(8)
    for (int i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    end = omp_get_wtime();
    printf("Parallel Pi:   %.10f, Time: %f sec\n", pi, end - start);
    printf("Speedup: %fx\n", seq_time / (end - start));

    return 0;
}