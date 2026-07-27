#include "flashvad_native.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef FV_BENCHMARK_ITERATIONS
#define FV_BENCHMARK_ITERATIONS 10000
#endif

#ifndef FV_INIT_ITERATIONS
#define FV_INIT_ITERATIONS 500
#endif

static int compare_double(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double elapsed_microseconds(
    uint64_t start,
    uint64_t end,
    mach_timebase_info_data_t timebase
) {
    const double nanoseconds =
        (double)(end - start) * (double)timebase.numer / (double)timebase.denom;
    return nanoseconds / 1000.0;
}

int main(void) {
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);

    FlashVadState state;
    double init_timings[FV_INIT_ITERATIONS];
    for (size_t index = 0; index < FV_INIT_ITERATIONS; ++index) {
        const uint64_t init_start = mach_continuous_time();
        if (flashvad_init(&state) != 0) {
            fputs("native initialization failed\n", stderr);
            return 1;
        }
        init_timings[index] = elapsed_microseconds(
            init_start,
            mach_continuous_time(),
            timebase
        );
        flashvad_destroy(&state);
    }
    qsort(init_timings, FV_INIT_ITERATIONS, sizeof(double), compare_double);

    if (flashvad_init(&state) != 0) {
        fputs("native initialization failed\n", stderr);
        return 1;
    }

    float hop[FV_HOP_SAMPLES] = {0};
    for (size_t index = 0; index < 1000; ++index) {
        (void)flashvad_process_hop(&state, hop);
    }
    flashvad_reset(&state);

    double *timings = malloc(FV_BENCHMARK_ITERATIONS * sizeof(double));
    if (timings == NULL) {
        flashvad_destroy(&state);
        return 1;
    }
    float probability = 0.0f;
    for (size_t index = 0; index < FV_BENCHMARK_ITERATIONS; ++index) {
        const uint64_t started = mach_continuous_time();
        probability = flashvad_process_hop(&state, hop);
        timings[index] = elapsed_microseconds(
            started,
            mach_continuous_time(),
            timebase
        );
    }
    qsort(timings, FV_BENCHMARK_ITERATIONS, sizeof(double), compare_double);
    const size_t median_index = (FV_BENCHMARK_ITERATIONS - 1) / 2;
    const size_t p95_index = (size_t)((FV_BENCHMARK_ITERATIONS - 1) * 0.95);
    const size_t p99_index = (size_t)((FV_BENCHMARK_ITERATIONS - 1) * 0.99);

    printf(
        "{\n"
        "  \"runtime\": \"accelerate-embedded\",\n"
        "  \"iterations\": %d,\n"
        "  \"hop_ms\": %.3f,\n"
        "  \"init_median_us\": %.3f,\n"
        "  \"init_p95_us\": %.3f,\n"
        "  \"init_p99_us\": %.3f,\n"
        "  \"combined_median_us\": %.3f,\n"
        "  \"combined_p95_us\": %.3f,\n"
        "  \"combined_p99_us\": %.3f,\n"
        "  \"realtime_factor\": %.9f,\n"
        "  \"last_probability\": %.9f\n"
        "}\n",
        FV_BENCHMARK_ITERATIONS,
        1000.0 * (double)FV_HOP_SAMPLES / (double)FV_SAMPLE_RATE,
        init_timings[(FV_INIT_ITERATIONS - 1) / 2],
        init_timings[(size_t)((FV_INIT_ITERATIONS - 1) * 0.95)],
        init_timings[(size_t)((FV_INIT_ITERATIONS - 1) * 0.99)],
        timings[median_index],
        timings[p95_index],
        timings[p99_index],
        timings[median_index]
            / (1000000.0 * (double)FV_HOP_SAMPLES / (double)FV_SAMPLE_RATE),
        probability
    );

    free(timings);
    flashvad_destroy(&state);
    return 0;
}
