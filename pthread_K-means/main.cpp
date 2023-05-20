#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <pthread.h>

#define MAX_ITERATIONS 1000
#define NUM_POINTS 10000
#define DIMENSIONS 8
#define NUM_CLUSTERS 100
#define NUM_THREADS 16  // 设置使用的线程数

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

// 定义线程函数
void* find_nearest_center_thread(void* thread_id) {
    int tid = *((int*) thread_id);
    int start = tid * (NUM_POINTS / NUM_THREADS);
    int end = (tid == NUM_THREADS - 1) ? NUM_POINTS : start + (NUM_POINTS / NUM_THREADS);

    for (int i = start; i < end; i++) {
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

    pthread_exit(NULL);
}

// 定义线程函数
void* update_centers_thread(void* thread_id) {
    int tid = *((int*) thread_id);
    int start = tid * (NUM_CLUSTERS / NUM_THREADS);
    int end = (tid == NUM_THREADS - 1) ? NUM_CLUSTERS : start + (NUM_CLUSTERS / NUM_THREADS);

    for (int i = start; i < end; i++) {
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

    pthread_exit(NULL);
}


int find_nearest_center(float *point) {
    int nearest_center = 0;
    float min_distance = INFINITY;
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        __m256 distance_avx = _mm256_setzero_ps();
        // float distance = 0.0;
        for (int j = 0; j < DIMENSIONS; j+=8){
            __m256 point_avx = _mm256_loadu_ps(point+j);
            __m256 center_avx = _mm256_loadu_ps(centers[i]+j);
            __m256 diff_avx = _mm256_sub_ps(point_avx, center_avx);
            __m256 squard_diff_avx = _mm256_mul_ps(diff_avx, diff_avx);
            distance_avx = _mm256_add_ps(distance_avx, squard_diff_avx);
        }

        float distance_sum = _mm256_cvtss_f32(distance_avx);
        if(distance_sum < min_distance){
            nearest_center=i;
            min_distance=distance_sum;
        }
    }
    return nearest_center;
}

void update_centers(int *cluster_assignments) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        __m256 sum_avx[DIMENSIONS/8];
        for (int j=0;j<DIMENSIONS/8;j++){
            sum_avx[j]=_mm256_setzero_ps();
        }
        int count = 0;
        for (int j = 0; j < NUM_POINTS; j++) {
            if (cluster_assignments[j] == i) {
                count++;
                for (int k = 0; k < DIMENSIONS; k+=8) {
                    __m256 points_avx = _mm256_loadu_ps(points[i]+k);
                    sum_avx[k/8]=_mm256_add_ps(sum_avx[k/8], points_avx);
                }
            }
        }
        if (count > 0) {
            for (int k = 0; k < DIMENSIONS; k+=8) {
                __m256 count_avx = _mm256_set1_ps(count);
                __m256 avg_avx = _mm256_div_ps(sum_avx[k/8],count_avx);
                _mm256_store_ps(centers[i]+k, avg_avx);
            }
        }
    }
}


int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    srand(time(0));
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            points[i][j] = (float) rand() / RAND_MAX;
        }
    }

    init_centers();

    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        // 创建并行线程来计算每个数据点的最近聚类中心

        for (int t = 0; t < NUM_THREADS; t++) {
            thread_ids[t] = t;
            pthread_create(&threads[t], NULL, find_nearest_center_thread, (void*) &thread_ids[t]);
        }

        // 等待所有线程完成
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }


        /*for (int i = 0; i < NUM_POINTS; i++) {
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
        }*/


        // 创建并行线程来更新聚类中心

        for (int t = 0; t < NUM_THREADS; t++) {
            thread_ids[t] = t;
            pthread_create(&threads[t], NULL, update_centers_thread, (void*) &thread_ids[t]);
        }

        // 等待所有线程完成
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }




        /*for (int i = 0; i < NUM_CLUSTERS; i++) {
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
        }*/


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
