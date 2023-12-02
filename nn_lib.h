#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "./mat_lib.h"

// Macros
#define MAT_AT(m, i, j) (m.data[i*m.stride + j])

// Header declarations
#ifndef NN_LIB

// Struct declarations
typedef struct Network{

	/* Architecture of the network */
	int layer_count; // This does not include the input layer
	int* arch; 

	/* Activations, neuron_inputs (activation but without applying the activation function), weights, biases */
	Mat input_layer;
	Mat* as; 
	Mat* zs;
	Mat* ws;
	Mat* bs;

} Network;

// Functions declarations
Network model_alloc(int* arch, int arch_count);
void model_init(Network N);
void model_rand(Network N);
void model_print(Network N);
void forward(Network N);
float cost_singular(Network N, Mat X, Mat Y);
float cost_total(Network N, Mat* Xs, Mat* Ys, int count);
void backprop(Network N, Mat* nabla_W, Mat* nabla_B, Mat X, Mat Y);
void model_train(Network N, Mat* Xs, Mat* Ys, int dataset_size, int epoch_count, float learn_rate);

#endif // !NN_LIB
       
// Library implementation
#ifdef NN_LIB_IMPLEMENTATION

Network model_alloc(int* arch, int arch_count) {
	Network N = {.layer_count = arch_count - 1,.arch = arch};

	N.as = (Mat*)malloc(sizeof(Mat) * N.layer_count);
	N.zs = (Mat*)malloc(sizeof(Mat) * N.layer_count);
	N.ws = (Mat*)malloc(sizeof(Mat) * N.layer_count);
	N.bs = (Mat*)malloc(sizeof(Mat) * N.layer_count);

	// Input layer
	N.input_layer = mat_alloc(arch[0], 1);

	for (int i = 0; i < N.layer_count; i++){
		N.zs[i] = mat_alloc(arch[i+1], 1);
		N.as[i] = mat_alloc(arch[i+1], 1);
		N.ws[i] = mat_alloc(arch[i+1], arch[i]);
		N.bs[i] = mat_alloc(arch[i+1], 1);
	}
	return N;
}

void model_init(Network N){
	mat_fill(N.input_layer, 0);
	for (int i = 0; i < N.layer_count; i++){
		mat_fill(N.as[i], 0);
		mat_fill(N.zs[i], 0);
	}
}

void model_rand(Network N){
	for (int i = 0; i < N.layer_count; i++){
		mat_rand(N.ws[i], -1, 1);
		mat_rand(N.bs[i], -1, 1);
	}
}

void model_print(Network N){
	printf("The input layer: \n");
	mat_print(N.input_layer);
	for (int i = 0; i < N.layer_count; i++){
		printf("W[%d]\n", i);
		mat_print(N.ws[i]);
		printf("B[%d]\n", i);
		mat_print(N.bs[i]);
		printf("z[%d]\n", i);
		mat_print(N.zs[i]);
		printf("a[%d]\n", i);
		mat_print(N.as[i]);
	}
}

void forward(Network N){
	mat_mul(N.zs[0], N.ws[0], N.input_layer);
	mat_add(N.zs[0], N.bs[0]);
	mat_map(N.as[0], N.zs[0], sigmoidf);
	for (int i = 1; i < N.layer_count; i++){
		mat_mul(N.zs[i], N.ws[i], N.as[i-1]);
		mat_add(N.zs[i], N.bs[i]);
		mat_map(N.as[i], N.zs[i], sigmoidf);
	}
}

float cost_singular(Network N, Mat X, Mat Y){
	float cost_sum = 0;
	Mat cost_vector = mat_alloc(Y.rows, 1);
	assert(N.as[N.layer_count - 1].rows == Y.rows);
	assert(N.input_layer.rows == X.rows);
	mat_copy(N.input_layer, X);
	forward(N);

	mat_copy(cost_vector, Y);
	mat_sub(cost_vector, N.as[N.layer_count - 1]);
	mat_map(cost_vector, cost_vector, squaref);

	for (int i = 0; i < cost_vector.rows; i++){
		cost_sum += MAT_AT(cost_vector, i, 0);
	}

	return cost_sum;
}

float cost_total(Network N, Mat* Xs, Mat* Ys, int count){
	// Total cost over all training (input, output) pairs
	float total_cost = 0;
	for (int i = 0; i < count; i++){
		total_cost += cost_singular(N, Xs[i], Ys[i]);
	}
	return total_cost;
}

// X is the input
// Y is the expected output
void backprop(Network N, Mat* nabla_W, Mat* nabla_B, Mat X, Mat Y){
	int temp_int = 0;
	int nodes_largest_layer = 0;
	for (int i = 1; i < N.layer_count + 1; i++){
		nodes_largest_layer = N.arch[i] > nodes_largest_layer ? N.arch[i] : nodes_largest_layer;
	}
	Mat delta = mat_alloc(nodes_largest_layer, 1);
	Mat delta_new = mat_alloc(nodes_largest_layer, 1);
	delta.rows = N.as[N.layer_count - 1].rows;
	delta_new.rows = N.as[N.layer_count - 1].rows;
	float* temp;

	mat_fill(delta, 0);
	mat_fill(delta_new, 0);

	mat_copy(N.input_layer, X);
	forward(N);

	// Calculating delta_L
	for (int j = 0; j < N.zs[N.layer_count - 1].rows; j++){
		// 2 * sigmoid_prime(z_L_j) * (a_L_j - y_j)
		MAT_AT(delta_new, j, 0) = 2 * prime_sigmoidf(MAT_AT(N.zs[N.layer_count - 1], j, 0)) * (MAT_AT(N.as[N.layer_count - 1], j, 0) - MAT_AT(Y, j, 0));
	}

	for (int l = N.layer_count - 1; l > 0; l--){
		// delta_new -> delta
		temp = delta.data;
		delta.data = delta_new.data;
		delta_new.data = temp;

		delta.rows = delta_new.rows;

		// Weights
		mat_transpose(&N.as[l-1]);
		mat_mul(nabla_W[l], delta, N.as[l-1]);
		mat_transpose(&N.as[l-1]);

		// Biases
		mat_copy(nabla_B[l], delta);

		// delta_new
		delta_new.rows = N.zs[l-1].rows;
		mat_transpose(&delta);
		mat_transpose(&delta_new);
		mat_mul(delta_new, delta, N.ws[l]);
		mat_transpose(&delta);
		mat_transpose(&delta_new);
		//THE DIFFERENCE/PROBLEM IS SOMEWHERE HERE!!!
		for (int row_idx = 0; row_idx < N.zs[l-1].rows; row_idx++){
			MAT_AT(delta_new, row_idx, 0) *= prime_sigmoidf(MAT_AT(N.zs[l-1], row_idx, 0));
		}
	}
	// Repeating everything for the first layer (index 0)
	temp = delta.data;
	delta.data = delta_new.data;
	delta_new.data = temp;

	delta.rows = delta_new.rows;

	mat_transpose(&N.input_layer);
	mat_mul(nabla_W[0], delta, N.input_layer);
	mat_transpose(&N.input_layer);

	// Biases
	mat_copy(nabla_B[0], delta);

	free(delta.data);
	free(delta_new.data);
}

void model_train(Network N, Mat* Xs, Mat* Ys, int dataset_size, int epoch_count, float learn_rate){
	// These are for the whole epoch
	Mat total_nabla_ws[N.layer_count];
	Mat total_nabla_bs[N.layer_count];
	// These are for individual training example
	Mat nabla_ws[N.layer_count];
	Mat nabla_bs[N.layer_count];

	for (int l = 0; l < N.layer_count; l++){
		total_nabla_ws[l] = mat_alloc(N.ws[l].rows, N.ws[l].cols);
		nabla_ws[l]       = mat_alloc(N.ws[l].rows, N.ws[l].cols);
		total_nabla_bs[l] = mat_alloc(N.bs[l].rows, 1);
		nabla_bs[l]       = mat_alloc(N.bs[l].rows, 1);
	}

	int data_idx[dataset_size];

	for (int i = 0; i < dataset_size; i++){
		data_idx[i] = i;
	}

	for (int epoch = 0; epoch < epoch_count; epoch++){

		// We need to randomize the order in which we feed the (INPUT, OUTPUT) pairs to the model
		rand_permute_array(data_idx, dataset_size);

		for (int i = 0; i < dataset_size; i++){
			backprop(N, nabla_ws, nabla_bs, Xs[data_idx[i]], Ys[data_idx[i]]);

			for (int l = 0; l < N.layer_count; l++){
				mat_scale(nabla_ws[l], nabla_ws[l], learn_rate);
				mat_scale(nabla_bs[l], nabla_bs[l], learn_rate);

				mat_sub(N.ws[l], nabla_ws[l]);
				mat_sub(N.bs[l], nabla_bs[l]);
			}
		}

		// Only for info
		printf("Total cost on epoch %d : %f\n", epoch, cost_total(N, Xs, Ys, dataset_size));
	}
}

#endif // NN_LIB_IMPLEMENTATION
