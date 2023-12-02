#pragma once

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Macros
#define MAT_AT(m, i, j) (m.data[i*m.stride + j])

// Header declarations
#ifndef MAT_LIB
#define MAT_LIB

// Struct declarations
typedef struct Mat{
	int rows;
	int cols;
	// Stride basically means how many elements you need to jump to get to the element with same index in the next row
	int stride;
	float* data;
} Mat;

// Function declarations
float rand_float();
float reLU(float x);
float squaref(float x);
float sigmoidf(float x);
float prime_sigmoidf(float x);
long random_at_most(long max);
void rand_permute_array(int* A, size_t n);
int get_max_element(Mat A);

Mat mat_alloc(int rows, int cols);
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat A, float x);
void mat_rand(Mat A, float low, float high);
Mat get_row(Mat src, int row_idx);
Mat get_col(Mat src, int col_idx);

Mat* mat_transpose(Mat* A);
void mat_add(Mat dst, Mat src);
void mat_sub(Mat dst, Mat src);
void mat_scale(Mat dst, Mat src, float scale);
void mat_mul(Mat dst, Mat A, Mat B);
void mat_map(Mat dst, Mat A, float (*f)(float x));
void mat_print(Mat A);

#endif // !MAT_LIB
 
#define MAT_LIB_IMPLEMENTATION
// Library implementation
#ifdef MAT_LIB_IMPLEMENTATION

// From 0 to 1
float rand_float(){
	float x = rand() / ((float) RAND_MAX);
	return x;
}

float reLU(float x){
	return x > 0 ? x : 0;
}

float squaref(float x){
	return x*x;
}

float sigmoidf(float x){
	return 1.f / (1.f + expf(-x));
}

float prime_sigmoidf(float x){
	return sigmoidf(x)/(1 + expf(x));
}

long random_at_most(long max) {
	unsigned long
		// max <= RAND_MAX < ULONG_MAX, so this is okay.
		num_bins = (unsigned long) max + 1,
			 num_rand = (unsigned long) RAND_MAX + 1,
			 bin_size = num_rand / num_bins,
			 defect   = num_rand % num_bins;

	long x;
	do {
		x = rand();
	}
	// This is carefully written not to overflow
	while (num_rand - defect <= (unsigned long)x);

	// Truncated division is intentional
	return x/bin_size;
}

void rand_permute_array(int* A, size_t n){
	int j, temp;
	for (int i = 0; i < n; i++){
		j = random_at_most(n - i - 1) + i;
		temp = A[i];
		A[i] = A[j];
		A[j] = temp;
	}
}

int get_max_element(Mat A){
	int max_element = MAT_AT(A, 0, 0);
	for (int i = 1; i < A.rows; i++){
		for (int j = 1; j < A.cols; j++){
			max_element = (max_element > MAT_AT(A, i, j)) ? max_element : MAT_AT(A, i, j);
		}
	}
	return max_element;
}

Mat mat_alloc(int rows, int cols){
	float* data = (float*)malloc(rows*cols*sizeof(float));
	Mat A = {.rows = rows, .cols = cols, .stride = cols, .data = data};
	return A;
}

void mat_copy(Mat dst, Mat src){
	assert(dst.rows*dst.cols == src.rows*src.cols);
	memcpy(dst.data, src.data, sizeof(src.data[0]) * src.rows * src.stride);
}

// Fills all the entries of the matrix with x
void mat_fill(Mat A, float x){
	for (int i = 0; i < A.cols; i++){
		for (int j = 0; j < A.cols; j++){
			MAT_AT(A, i, j) = x;
		}
	}
}

void mat_rand(Mat A, float low, float high){
	for (int i = 0; i < A.rows; i++){
		for (int j = 0; j < A.cols; j++){
			MAT_AT(A, i, j) = (high - low)*rand_float() + low;
		}
	}
}

Mat get_row(Mat src, int row_idx){
	assert(row_idx < src.rows);
	assert(-1 < row_idx);
	Mat result = {.rows = 1, .cols = src.cols, .stride = src.stride, .data = (src.data + src.stride*row_idx)};
	return result;
};

Mat get_col(Mat src, int col_idx){
	assert(col_idx < src.cols);
	assert(-1 < col_idx);
	Mat result = {.rows = src.rows, .cols = 1, .stride = src.stride, .data = (src.data + col_idx)};
	return result;
};

/*Extremely slow, needs to be rewritten*/
Mat* mat_transpose(Mat* A){
	Mat A_t = mat_alloc(A->cols, A->rows);
	for (int i = 0; i < A->rows; i++){
		for (int j = 0; j < A->cols; j++){
			MAT_AT(A_t, j, i) = (A->data[i*A->stride + j]);
		}
	}
	mat_copy(*A, A_t);
	A->cols = A_t.cols;
	A->rows = A_t.rows;
	A->stride = A_t.cols;
	return A;
}

void mat_add(Mat dst, Mat src){
	assert(dst.cols == src.cols);
	assert(dst.rows == src.rows);
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			MAT_AT(dst, i, j) += MAT_AT(src, i, j);
		}
	}
}

void mat_sub(Mat dst, Mat src){
	assert(dst.cols == src.cols);
	assert(dst.rows == src.rows);
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			MAT_AT(dst, i, j) -= MAT_AT(src, i, j);
		}
	}
}

void mat_scale(Mat dst, Mat src, float scale){
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			MAT_AT(dst, i, j) = scale*MAT_AT(src, i, j);
		}
	}
}

void mat_entrywise_mul(Mat dst, Mat A, Mat B){
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			MAT_AT(dst, i, j) = MAT_AT(A, i, j)*MAT_AT(B, i, j);
		}
	}
}

void mat_mul(Mat dst, Mat A, Mat B){
	assert(A.cols == B.rows);
	assert(dst.rows == A.rows);
	assert(dst.cols == B.cols);
	float temp_sum = 0;
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			temp_sum = 0;
			for (int k = 0; k < A.cols; k++){
				temp_sum += MAT_AT(A, i, k) * MAT_AT(B, k, j);
			}
			MAT_AT(dst, i, j) = temp_sum;
		}
	}
}

void mat_map(Mat dst, Mat A, float (*f)(float x)){
	for (int i = 0; i < A.rows; i++){
		for (int j = 0; j < A.cols; j++){
			MAT_AT(dst, i, j) = f(MAT_AT(A, i, j));
		}
	}
}

void mat_print(Mat A){
	printf("[\n");
	for (int i = 0; i < A.rows; i++){
		for (int j = 0; j < A.cols; j++){
			printf(" %f", MAT_AT(A, i, j));
		}
		printf(" \n");
	}
	printf("]\n");
}

#endif // MAT_LIB_IMPLEMENTATION
