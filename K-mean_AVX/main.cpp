#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>

#define MAX_ITERATIONS 100
#define NUM_POINTS 1000
#define DIMENSIONS 8
#define NUM_CLUSTERS 100

using namespace std;
float points[NUM_POINTS][DIMENSIONS];
float centers[NUM_CLUSTERS][DIMENSIONS];
void init_centers() {
    srand(time(0));
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centers[i][j] = (float) rand() / RAND_MAX;
        }
    }
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
    srand(time(0));
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            points[i][j] = (float) rand() / RAND_MAX;
            //cout<<points[i][j]<<" ";
        }
        //cout<<")"<<endl;
    }

    init_centers();
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        int cluster_assignments[NUM_POINTS];
        for (int i = 0; i < NUM_POINTS; i++) {
            cluster_assignments[i] = find_nearest_center(points[i]);
        }

        update_centers(cluster_assignments);
    }


    // Print the final cluster centers

    /*cout << "Final cluster centers:\n";
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        cout << "Cluster " << i << ": (";
        for (int j = 0; j < DIMENSIONS; j++)
            cout << centers[i][j] << " ";
        cout << ")\n";
    }*/
    return 0;
}
