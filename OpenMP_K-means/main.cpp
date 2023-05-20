#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <omp.h>

#define MAX_ITERATIONS 1000
#define NUM_POINTS 10000
#define DIMENSIONS 8
#define NUM_CLUSTERS 100
#define NUM_THREADS 16 // 设置使用的线程数

using namespace std;

float points[NUM_POINTS][DIMENSIONS];
float centers[NUM_CLUSTERS][DIMENSIONS];
int cluster_assignments[NUM_POINTS];

void init_centers() {
    srand(time(0));
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centers[i][j] = (float) rand() / RAND_MAX;
        }
    }
}

void find_nearest_center() {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < NUM_POINTS; i++) {
        int nearest_center = 0;
        float min_distance = INFINITY;
        for (int j = 0; j < NUM_CLUSTERS; j++) {
            __m256 distance_avx = _mm256_setzero_ps();
            for (int k = 0; k < DIMENSIONS; k+=8){
                __m256 point_avx = _mm256_loadu_ps(points[i]+k);
                __m256 center_avx = _mm256_loadu_ps(centers[j]+k);
                __m256 diff_avx = _mm256_sub_ps(point_avx, center_avx);
                __m256 squared_diff_avx = _mm256_mul_ps(diff_avx, diff_avx);
                distance_avx = _mm256_add_ps(distance_avx, squared_diff_avx);
            }

            float distance_sum = _mm256_cvtss_f32(distance_avx);
            if (distance_sum < min_distance) {
                nearest_center = j;
                min_distance = distance_sum;
            }
        }
        cluster_assignments[i] = nearest_center;
    }
}

void update_centers() {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        __m256 sum_avx[DIMENSIONS/8];
        for (int j = 0; j < DIMENSIONS/8; j++) {
            sum_avx[j] = _mm256_setzero_ps();
        }
        int count = 0;

        for (int j = 0; j < NUM_POINTS; j++) {
            if (cluster_assignments[j] == i) {
                count++;
                for (int k = 0; k < DIMENSIONS; k+=8) {
                    __m256 points_avx = _mm256_loadu_ps(points[j]+k);
                    sum_avx[k/8] = _mm256_add_ps(sum_avx[k/8], points_avx);
                }
            }
        }

        if (count > 0) {
            for (int k = 0; k < DIMENSIONS; k+=8) {
                __m256 count_avx = _mm256_set1_ps(count);
                __m256 avg_avx = _mm256_div_ps(sum_avx[k/8], count_avx);
                _mm256_store_ps(centers[i]+k, avg_avx);
            }
        }
    }
}

int main() {
    srand(time(0));
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            points[i][j] = (float) rand() / RAND_MAX;
        }
    }

    init_centers();

    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        find_nearest_center();
        update_centers();
    }

    // 打印最终聚类中心
    /*cout << "Final cluster centers:\n";
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        cout << "Cluster " << i << ": (";
        for (int j = 0; j < DIMENSIONS; j++)
            cout << centers[i][j] << " ";
        cout << ")\n";
    }*/

    return 0;
}
