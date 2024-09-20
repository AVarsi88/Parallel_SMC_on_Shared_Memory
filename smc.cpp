#include <vector>
#include <random>
#include "importance_sampling.h"
#include "reduction_operations.h"
#include "resampling.h"
#include "model.h"
#include <cmath>
#include <omp.h>

double calculateStandardDeviation(std::vector<double>& data) {
	int n = data.size();
	if (n <= 1) {
		// If the vector has fewer than 2 elements, return 0 since standard deviation is undefined.
		return 0.0;
	}

	// Calculate the mean (average) of the elements in the vector.
	double sum = 0.0;
	for (double val : data) {
		sum += val;
	}
	double mean = sum / n;

	// Calculate the sum of squares of differences between each element and the mean.
	double sumSquaredDiff = 0.0;
	for (double val : data) {
		double diff = val - mean;
		sumSquaredDiff += diff * diff;
	}

	// Calculate the variance (average of the sum of squares of differences).
	double variance = sumSquaredDiff / n;

	// Standard deviation is the square root of the variance.
	double stdDeviation = std::sqrt(variance);

	return stdDeviation;
}

double smc_sampler(int N, int K, int T, int redistribution, double *time, double *red_percentage, int seed = 0) {
	std::vector<std::vector<int>> x(N);
	std::vector<double> logw(N);
	std::vector<int> initM(N);
	std::vector<std::vector<int>> initx(N);
	double neff, Nt = (double)(N), *time_resampling = new double;

	*time_resampling = 0.0;

	std::vector<std::mt19937> mts(N);
	std::mt19937 mt_resampling;
	mt_resampling.seed(seed);

	std::vector<double> sizes(N);

	initialise(x, logw, initx, initM, N, T, seed, mts);

	/*printf("x = \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < x[i].size(); j++) {
			printf("%d ", x[i][j]);
		}
		printf("\n");
	}*/
        
        double start = omp_get_wtime();
        
	for (int k = 0; k < K; k++) {
		/*printf("logw before normalisation = [");
		for (int i = 0; i < N; i++)
			printf("%lf ", logw[i]);
		printf("]\n");*/
		normalise(logw, N, T);
		/*printf("logw after normalisation = [");
		for (int i = 0; i < N; i++)
			printf("%lf ", logw[i]);
		printf("]\n");*/


		neff = ESS(logw, N, T);
		//printf("ESS = %lf\n", neff);

		if (neff < Nt) {
			//printf("I'm here!!!\n");
			resampling(x, logw, N, T, mt_resampling, redistribution, time_resampling);
		}
		//printf("post resampling!!!\n");
		importance_sampling(x, logw, initx, initM, N, T, mts);

		/*printf("logw after IS = [");
		for (int i = 0; i < N; i++)
			printf("%lf ", logw[i]);
		printf("]\n");*/

		/*int num_big_samples = 0;
		for (int i = 0; i < N; i++) {
			sizes.at(i) = x[i].size();
			if (sizes.at(i) > get_mid_size()) {
				num_big_samples++;
			}
		}
		printf("percentage of big samples = %lf\n", (double)num_big_samples / N);

		printf("STD of sizes: %lf\n", calculateStandardDeviation(sizes)); */
                printf("Iteration %d\n", k);
	}
        
        double end = omp_get_wtime();
        
        *time = end - start;

		//printf("time resampling = %lf\n", *time_resampling / *time);

		*red_percentage = *time_resampling / *time;
	
	return estimate(x, logw, N, T, f);
}