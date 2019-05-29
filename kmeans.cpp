////////////////
// 
// File: kmeans.cpp
//
//  Main body of K-Means simulaton. Reads in the original data points from
//  `ori.txt`, performs K-Means clustering on randomly-picked initial
//  centers, and writes the results into `res.txt` with the same format.
//
//  * You may (and should) include some extra headers for optimizations.
//
//  * You should and ONLY should modify the function body of `kmeans()`.
//    DO NOT change any other exitsing part of the program.
//
//  * You may add your own auxiliary functions if you wish. Extra declarations
//    can go in `kmeans.h`.
//
// Jose @ ShanghaiTech University
//
////////////////

#include <fstream>
#include <limits>
#include <math.h>
#include <chrono>
#include "kmeans.h"


/*********************************************************
        Your extra headers and static declarations
 *********************************************************/
#include <cassert>
#include <omp.h>
/*********************************************************
                           End
 *********************************************************/


/*
 * Entrance point. Time ticking will be performed, so it will be better if
 *   you have cleared the cache for precise profiling.
 *
 */
int
main (int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input.txt> <output.txt>"
                  << std::endl;
        return -1;
    }
    if (!(bool)std::ifstream(argv[1])) {
        std::cerr << "ERROR: Data file " << argv[1] << " does not exist!"
                  << std::endl;
        return -1;
    }
    if ((bool)std::ifstream(argv[2])) {
        std::cerr << "ERROR: Destination " << argv[2] << " already exists!"
                  << std::endl;
        return -1;
    }
    FILE *fi = fopen(argv[1], "r"), *fo = fopen(argv[2], "w");
    
    /* From `ori.txt`, acquire dataset size, number of colors (i.e. K in
       K-Means),and read in all data points into static array `data`. */
    int pn, cn;

    assert(fscanf(fi, "%d / %d\n", &pn, &cn) == 2);

    point_t * const data = new point_t[pn];
    color_t * const coloring = new color_t[pn];

    for (int i = 0; i < pn; ++i)
        coloring[i] = 0;

    int i = 0, c;
    double x, y;

    while (fscanf(fi, "%lf, %lf, %d\n", &x, &y, &c) == 3) {
        data[i++].setXY(x, y);
        if (c < 0 || c >= cn) {
            std::cerr << "ERROR: Invalid color code encoutered!"
                      << std::endl;
            return -1;
        }
    }
    if (i != pn) {
        std::cerr << "ERROR: Number of data points inconsistent!"
                  << std::endl;
        return -1;
    }

    /* Generate a random set of initial center points. */
    point_t * const mean = new point_t[cn];

    srand(5201314);
    for (int i = 0; i < cn; ++i) {
        int idx = rand() % pn;
        mean[i].setXY(data[idx].getX(), data[idx].getY());
    }

    /* Invode K-Means algorithm on the original dataset. It should cluster
       the data points in `data` and assign their color codes to the
       corresponding entry in `coloring`, using `mean` to store the center
       points. */
    std::cout << "Doing K-Means clustering on " << pn
              << " points with K = " << cn << "..." << std::flush;
    auto ts = std::chrono::high_resolution_clock::now();
    kmeans(data, mean, coloring, pn, cn);
    auto te = std::chrono::high_resolution_clock::now();
    std::cout << "done." << std::endl;
    std::cout << " Total time elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds> \
                 (te - ts).count()
              << " milliseconds." << std::endl; 

    /* Write the final results to `res.txt`, in the same format as input. */
    fprintf(fo, "%d / %d\n", pn, cn);
    for (i = 0; i < pn; ++i)
        fprintf(fo, "%.8lf, %.8lf, %d\n", data[i].getX(), data[i].getY(),
                coloring[i]);

    /* Free the resources and return. */
    delete[](data);
    delete[](coloring);
    delete[](mean);
    fclose(fi);
    fclose(fo);
    return 0;
}


/*********************************************************
           Feel free to modify the things below
 *********************************************************/

/*
 * K-Means algorithm clustering. Originally implemented in a traditional
 *   sequential way. You should optimize and parallelize it for a better
 *   performance. Techniques you can use include but not limited to:
 *
 *     1. OpenMP shared-memory parallelization.
 *     2. SSE SIMD instructions.
 *     3. Cache optimizations.
 *     4. Manually using pthread.
 *     5. ...
 *
 */
void
kmeans (point_t * const data, point_t * const mean, color_t * const coloring,
        const int pn, const int cn)
{
    bool converge = true;

    // omp_set_num_threads(4);

    // std::cout << "Run with " << omp_get_max_threads() << "threads" << std::endl;

    /* Loop through the following two stages until no point changes its color
       during an iteration. */
    do {
        converge = true;

        double agg_sums_x[20] = {0.0f};
        double agg_sums_y[20] = {0.0f};
        int agg_counts[20] = {0};

        /* Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */
        #pragma omp parallel
        {
            double sums_x[20] = {0.0f};
            double sums_y[20] = {0.0f};
            int counts[20] = {0};
            bool local_converge = true;

            #pragma omp for nowait
            for (int i = 0; i < pn; ++i)
            {
                color_t new_color = cn;
                double min_dist = std::numeric_limits<double>::infinity();

                for (color_t c = 0; c < cn; ++c)
                {
                    double dist = pow(data[i].getX() - mean[c].getX(), 2) +
                                        pow(data[i].getY() - mean[c].getY(), 2);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        new_color = c;
                    }

                    // __m256d m_data_xy = _mm256_load_pd((double*)&data[i]);
                    // __m256d m_mean_xy = _mm256_broadcast_pd((__m128d*)(&mean[c]));
                    // __m256d m_dist_xy = _mm256_sub_pd(m_data_xy, m_mean_xy);
                    // m_dist_xy = _mm256_mul_pd(m_dist_xy, m_dist_xy);
                    // __m256d m_dists = _mm256_hadd_pd(m_dist_xy, m_dist_xy);
                }

                if (coloring[i] != new_color)
                {
                    coloring[i] = new_color;
                    local_converge = false;
                }
                sums_x[new_color] += data[i].getX();
                sums_y[new_color] += data[i].getY();
                counts[new_color]++;
            }

            // The follows are two-stage pipelining reduction
            #pragma omp critical
            for (color_t c = 0; c < cn; ++c)
            {
                agg_sums_x[c] += sums_x[c];
                agg_sums_y[c] += sums_y[c];
            }
            #pragma omp critical
            for (color_t c = 0; c < cn; ++c)
            {
                agg_counts[c] += counts[c];
                converge &= local_converge;
            }
            // yhj
        }

        /* Calculate the new mean for each cluster to be the current average
           of point positions in the cluster. */
        for (color_t c = 0; c < cn; ++c)
        {
            mean[c].setXY(agg_sums_x[c] / agg_counts[c], agg_sums_y[c] / agg_counts[c]);
        }

    } while (!converge);
}

/*********************************************************
                           End
 *********************************************************/
