
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdlib>

// 1. Apple Native Vector Math/BLAS Header
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

using namespace std;
using namespace chrono;

int main() {
    string input;

    int n;
    cout << "Enter matrix dimension (default = 4096): n = ";
    getline(std::cin, input);
    n = input.empty() ? 4096 : stoi(input);

    if (n <= 0) {
        cerr << "Error: n must be positive\n";
        return 1;
    }

    // 2. POSIX-compliant 64-byte aligned allocation for Apple Silicon
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;

    if (posix_memalign((void**)&A, 64, n * n * sizeof(double)) != 0 ||
        posix_memalign((void**)&B, 64, n * n * sizeof(double)) != 0 ||
        posix_memalign((void**)&C, 64, n * n * sizeof(double)) != 0) {
        cerr << "Memory allocation failed\n";
        return 1;
    }

    // Random initialization
    random_device rd;
    mt19937_64 rng(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t i = 0; i < (size_t)n * n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Warm-up (Apple AMX configuration handshake happens on first call)
    int warm = min(n, 512);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        warm, warm, warm,
        1.0, A, n, B, n, 0.0, C, n);

    int iterations;
    cout << "Enter number of iterations (default = 5): n = ";
    getline(std::cin, input);
    iterations = input.empty() ? 5 : stoi(input);
    vector<double> times(iterations);

    cout << "\nBenchmarking (" << iterations << " iterations):\n";
    cout << "========================================\n";

    for (int i = 0; i < iterations; ++i) {
        // Reset C
        std::fill(C, C + (size_t)n * n, 0.0);

        auto start = high_resolution_clock::now();

        // 3. Native cblas_dgemm utilizing the Apple Matrix Coprocessor
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, n, n,
            1.0, A, n, B, n, 0.0, C, n);

        auto end = high_resolution_clock::now();

        times[i] = duration<double>(end - start).count();
        cout << "  Iteration " << i + 1 << ": "
            << times[i] * 1000 << " ms ("
            << (2.0 * n * n * n / 1e9) / times[i] << " GFLOPS)\n";
    }

    sort(times.begin(), times.end());

    double median_time = times[iterations / 2];
    double best_time = times[0];

    double flops = 2.0 * n * n * n;
    double best_gflops = (flops / 1e9) / best_time;
    double median_gflops = (flops / 1e9) / median_time;

    cout << "\n=== RESULTS ===\n";
    cout << "Best time:    " << best_time * 1000 << " ms\n";
    cout << "Median time:  " << median_time * 1000 << " ms\n\n";
    cout << "Best GFLOPS:   " << best_gflops << "\n";
    cout << "Median GFLOPS: " << median_gflops << "\n";

    // 4. Clean up using standard free
    free(A);
    free(B);
    free(C);

    cout << "\nPress Enter to exit...";
    cin.ignore();
    cin.get();

    return 0;
}
