#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <windows.h>
#include <mpi.h>
#include <omp.h>

#define MAX_ITERATIONS 1000
#define NUM_POINTS 50000
#define DIMENSIONS 8
#define NUM_CLUSTERS 64
#define NUM_THREADS 16

using namespace std;

float points[NUM_POINTS][DIMENSIONS];
float centers[NUM_CLUSTERS][DIMENSIONS];
int cluster_assignments[NUM_POINTS];

void init_centers(float initial_centers[NUM_CLUSTERS][DIMENSIONS]) {
    srand(0);
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            centers[i][j] = initial_centers[i][j];
        }
    }
}

void generate_points(float points[NUM_POINTS][DIMENSIONS]) {
    //srand(0);
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < DIMENSIONS; j++) {
            points[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void find_nearest_center(int start, int end) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = start; i < end; i++) {
        int nearest_center = 0;
        float min_distance = numeric_limits<float>::max();  // 设置初始最小距离为较大值
        for (int j = 0; j < NUM_CLUSTERS; j++) {
            __m256 distance_avx = _mm256_setzero_ps();
            for (int k = 0; k < DIMENSIONS; k += 8) {
                __m256 point_avx = _mm256_loadu_ps(points[i] + k);
                __m256 center_avx = _mm256_loadu_ps(centers[j] + k);
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

void update_centers(int start, int end) {
    float local_centers[NUM_CLUSTERS][DIMENSIONS] = { 0 };
    int local_cluster_counts[NUM_CLUSTERS] = { 0 };

    for (int i = start; i < end; i++) {
        int cluster = cluster_assignments[i];
        for (int j = 0; j < DIMENSIONS; j++) {
            local_centers[cluster][j] += points[i][j];
        }
        local_cluster_counts[cluster]++;
    }

    MPI_Allreduce(MPI_IN_PLACE, local_cluster_counts, NUM_CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (local_cluster_counts[i] > 0) {
            for (int j = 0; j < DIMENSIONS; j++) {
                centers[i][j] = local_centers[i][j] / local_cluster_counts[i];
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, centers, NUM_CLUSTERS * DIMENSIONS, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
    QueryPerformanceCounter( (LARGE_INTEGER *)&head ) ;

    int num_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_points_per_process = NUM_POINTS / num_processes;
    int start = rank * num_points_per_process;
    int end = start + num_points_per_process;
    if (end > NUM_POINTS){
        end=NUM_POINTS;
    }

    if (rank == 0) {
        float initial_centers[NUM_CLUSTERS][DIMENSIONS];
        //srand(0);
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            for (int j = 0; j < DIMENSIONS; j++) {
                initial_centers[i][j] = (float)rand() / RAND_MAX;
            }
        }
        init_centers(initial_centers);  // 初始化聚类中心
        generate_points(points);  // 生成数据点
    }

    // 广播聚类中心给其他进程
    MPI_Bcast(centers, NUM_CLUSTERS * DIMENSIONS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 广播数据点给其他进程
    MPI_Bcast(points, NUM_POINTS * DIMENSIONS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        find_nearest_center(start, end);

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, cluster_assignments, num_points_per_process, MPI_INT,
            MPI_COMM_WORLD);

        update_centers(start, end);
    }

    /*if (rank == 0) {
        cout << "Final cluster centers:\n";
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            cout << "Cluster " << i << ": (";
            for (int j = 0; j < DIMENSIONS; j++)
                cout << centers[i][j] << " ";
            cout << ")\n";
        }
    }*/
    MPI_Finalize();
    QueryPerformanceCounter ( (LARGE_INTEGER *)&tail ) ;
    cout<<"Col : "<<( tail - head) * 1000.0 / freq<<"ms"<<endl;

    return 0;
}
