#include "Header_Lib.h"
#include <stdlib.h>

#define N 64

float scale(int i, int n){
	return ((float)i/(n-1)); //Cast i into 'float' data type!
}

int main(){
	
	const float ref = 0.5f; // Define the variable as the 'const'
	float *in = (float*)calloc(N, sizeof(float));
	float *out = (float*)calloc(N, sizeof(float));

	for (int i = 0; i < N; ++i){
		in[i] = scale(i, N);
	}

	distanceArray(out, in, ref, N);

	free(in);
	free(out);
	
	return 0;
}

//Coded by : Yehezk34