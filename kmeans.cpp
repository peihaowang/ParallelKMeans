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
#include <cstring>
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
    bool converge;

    /* Loop through the following two stages until no point changes its color
       during an iteration. */
    do {
        /* Conduct a distributive k-means algorithm, thus, allocate a t by c
           matrix to accommodate distributive results, where t is the number
           of threads, c is the number of centers. */
        int num_threads = omp_get_max_threads();
        // Accumulation for each partition
        point_t* sums = new point_t[num_threads * cn];
        // Counts of accumulation for each partition
        int* counts = new int[num_threads * cn];
        memset(counts, 0, num_threads * cn * sizeof(int));

        // Record the total number of threads
        int thread_num = 0;

        // Reset converge flag
        converge = true;

        /* Map: Compute the color of each point. A point gets assigned to the
           cluster with the nearest center point. */
        #pragma omp parallel
        {
            #pragma omp master
            thread_num = omp_get_num_threads();

            int id = omp_get_thread_num();

            // Note that, here reduce converge for each threads
            #pragma omp for reduction(&& : converge)
            for (int i = 0; i < pn; ++i)
            {
                color_t new_color = cn;
                double min_dist = std::numeric_limits<double>::infinity();

                for (color_t c = 0; c < cn; ++c)
                {
                    double dist = sqrt(pow(data[i].getX() - mean[c].getX(), 2) +
                                       pow(data[i].getY() - mean[c].getY(), 2));
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        new_color = c;
                    }
                }

                if (coloring[i] != new_color)
                {
                    // if(id == 0){
                    //     double dist = sqrt(pow(data[i].getX() - mean[coloring[i]].getX(), 2) +
                    //         pow(data[i].getY() - mean[coloring[i]].getY(), 2));
                    //     printf("Thread: %d, point %d, old %d(%lf), center %d(%lf)\n", id, i, coloring[i], dist, new_color, min_dist);
                    // }

                    coloring[i] = new_color;
                    converge = false;
                }

                // Now accumulate points for each centers
                int idx = id * cn + new_color;
                point_t& p = sums[idx];
                double x = p.getX() + data[i].getX();
                double y = p.getY() + data[i].getY();
                p.setXY(x, y);
                // Accumulate counts for reduction
                counts[idx]++;
            }
        }
        
        /* Reduction: Calculate the new mean for each cluster to be the
           current average of point positions in the cluster. */
        for (int i = 0; i < cn; i++)
        {
            point_t agg_sum;
            int agg_count = 0;

            for(int j = 0; j < thread_num; j++){
                int idx = j * cn + i;
                point_t& s = sums[idx];

                // Sum up all the summation of partition
                double x = s.getX() + agg_sum.getX();
                double y = s.getY() + agg_sum.getY();
                agg_sum.setXY(x, y);
                // Sum up all counts
                agg_count += counts[idx];
            }

            mean[i].setXY(
                agg_sum.getX() / agg_count
                , agg_sum.getY() / agg_count
            );
            // std::cout << "Mean:" << mean[i] << std::endl;
            // std::cout << "AGG:" << agg_sum << std::endl;
            // std::cout << "COUNT:" << agg_count << std::endl;
        }

        delete[] sums;
        delete[] counts;

    } while (!converge);
}

/*********************************************************
                           End
 *********************************************************/
