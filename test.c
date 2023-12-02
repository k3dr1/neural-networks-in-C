#include <stdio.h>
#include <time.h>
#include "mat_lib.h"

int main(){

	int sums[5] = {0};

	int A[5] = {0,1,2,3,4};
	for (int k = 0; k < 100000; k++){
		rand_permute_array(A, 5);

		for (int i = 0; i < 5; i++){
			//printf("%d ", A[i]);
			sums[i] += A[i];
		}
		//printf("\n");
	}

	for (int i = 0; i < 5; i++){
		printf("%d ", sums[i]);
	}
}
