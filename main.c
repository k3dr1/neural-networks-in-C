#define MAT_LIB_IMPLEMENTATION
#define NN_LIB_IMPLEMENTATION

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "./mat_lib.h"
#include "./nn_lib.h"

float train_data[][3] = {
	{0,0,0},
	{0,1,1},
	{1,0,1},
	{1,1,0},
};

int main (){

	srand(time(NULL));

	Mat Xs[4] = { 
		{.rows = 2, .cols = 1, .stride = 1, .data = (float*)(train_data)},
		{.rows = 2, .cols = 1, .stride = 1, .data = (float*)(train_data + 1)},
		{.rows = 2, .cols = 1, .stride = 1, .data = (float*)(train_data + 2)},
		{.rows = 2, .cols = 1, .stride = 1, .data = (float*)(train_data + 3)},
	};
	Mat Ys[4] = {
		{.rows = 1, .cols = 1, .stride = 1, .data = (float*)train_data[0] + 2},
		{.rows = 1, .cols = 1, .stride = 1, .data = (float*)train_data[1] + 2},
		{.rows = 1, .cols = 1, .stride = 1, .data = (float*)train_data[2] + 2},
		{.rows = 1, .cols = 1, .stride = 1, .data = (float*)train_data[3] + 2},
	};

	int xor_arch[] = {2,100,25,1};
	Network xor = model_alloc(xor_arch, sizeof(xor_arch)/sizeof(xor_arch[0]));
	model_init(xor);
	model_rand(xor);

	printf("XOR model:\n");
	//model_print(xor);

	// learning
	float learnRate = 0.5f;
	float cost_vector[4];
	float totalCost = 0;

	model_train(xor, Xs, Ys, 4, 10000, learnRate);

	for (int i = 0; i < 4; i++){
		cost_vector[i] = cost_singular(xor, Xs[i], Ys[i]);
		printf("Expected %f Actual %f Error %f\n", MAT_AT(Ys[i], 0, 0), MAT_AT(xor.as[xor.layer_count - 1], 0, 0), cost_vector[i]);
	}
	totalCost = cost_total(xor, Xs, Ys, 4);
	printf("Total cost after training: %f\n", totalCost);

	return 0;
}
