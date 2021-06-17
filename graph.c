#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

int check_color (int *color_arr, int N)
{
	int i;
	// if there is a vertex that is not colored
	for (i = 0; i < N; i++)
	{
		if (color_arr[i] < 1)
		{
			return 1;
		}
	}
	return 0;
}

float check_sparseness (float **Dense, int N)
{
	int i, j;
	int count = 0;
	// find the ratio to 0s to 1s in matrix
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (Dense[i][j] == 0)
			{
				count++;
			}
		}
	}
	float s = (float) count/(N*N);
	return s;
}

/* --------------------------------------------- CSR Functions ----------------------------------------------- */

void color_csr_graph (float *V, int *ROW, int *COL, int *color_arr, int N)
{	
	int color = 1;
	int i, j;
	// Initialize temp color
	int* temp_color= (int*)malloc(N*sizeof(int)); 
	for (i = 0; i < N; i++)
	{
		temp_color[i] = 0;
	}

	// While there was an uncolored vertex
	while (check_color (color_arr, N) == 1)
	{
		// Iterate over all vertices
		#pragma omp parallel for
		for (i = 0; i < N; i++)
		{
			if (temp_color[i] < 1)
			{
				float max_value = 0;
				float cur_value = V[i];
				// Iterate over the vertices that are connected by an edge
				#pragma omp parallel for reduction(max : max_value)
				for (j = ROW[i]; j < ROW[i+1]; j++)
				{
					// Find the maximum value from the neighbors
					if (temp_color[COL[j]] == 0 && V[COL[j]] > max_value)
					{
						max_value = V[COL[j]];
					}
				}
				// If the current vertex is uncolored and is the maximum value
				if (cur_value == max_value)
				{
					color_arr[i] = color;	
				}
			}
		}
	
		color++;
		// copy over the color to a temporary array so the colors checked are only from previous iteration
		#pragma omp parallel for
		for (i = 0; i < N; i++)
		{
			temp_color[i] = color_arr[i];
		}
	}
}


// Convert dense to csr matrix
void create_csr (float **Dense, int *COL, int *ROW, int N)
{
	int r = 0, c = 0;
	int i, j;
	// Row array begins with 0
	ROW[r] = 0;
	r++;
	
	// Iterate over rows - vertices
	for (i = 0; i < N; i++)
	{
		int ex = 0;
		for (j = 0; j < N; j++)
		{
			// If the element is a non-zero
			if (Dense[i][j] == 1)
			{
				ex = 1;
				// add to column array
				COL[c] = j;
				c++;
			}
		}
		// if there is any edge -- add to row array
		if (ex == 1)
		{
			ROW[r] = c;
			r++;
		}
	}
}

/* --------------------------------------------- Dense Functions ----------------------------------------------- */

float max_rand (float *V, int *color_arr,int N)
{
	float max_value = 0;
	int i;
	// find maximmum value in the row
	#pragma omp parallel for reduction(max : max_value)
	for (i = 0; i < N; i++)
	{
		if (V[i] > max_value && color_arr[i] == 0)
		{
			max_value = V[i];
		}
	}
	return max_value;
}

void color_graph (float **V, int *color_arr, int N)
{	
	int color = 1;
	int* temp_color= (int*)malloc(N*sizeof(int));  
	int i, j;
	while (check_color (color_arr, N) == 1)
	{
		#pragma omp parallel for 
		for (i = 0; i < N; i++)
		{
			// if the current vertex has the max value
			// then assign color
			float cur_value = V[i][i];
			float max_value = max_rand (V[i], temp_color, N);
			if (temp_color[i] < 1 && cur_value == max_value)
			{
				color_arr[i] = color;				
			}
		}
		color++;
		
		//#pragma omp parallel for 
		for (i = 0; i < N; i++)
		{
			temp_color[i] = color_arr[i];
		}
	}
}

/* ------------------------------------------------------------------------------------------------------- */

int main (int argc, char* argv[])
{
	int N = 5;
	int i, j;
	int arg_index = 1;
	
	double startTime, endTime;
	
	// Read optional command line arguments 
	while (arg_index < argc)
	{
		if ( strcmp(argv[arg_index], "-n") == 0 )
		{
			arg_index++;
			N = atoi(argv[arg_index++]);
		}
		
	}

	int E = N*N;

	// Arrays for CSR
	// Creating value, column, and row pointer array
	float* V = (float*)malloc(N*sizeof(float));   
	int* COL = (int*)malloc(E*sizeof(int));  
	int* ROW_PTR = (int*)malloc(N*sizeof(int)); 

	// Used to create different random values at each execution
	srand((unsigned)time(NULL));
	
	// create array of values with random number for each vertex
	for (i = 0; i < N; i++)
	{
		V[i] = (float)rand()/(float)RAND_MAX;
	
	}
	
	// Create dense matrix
	float **D;
	D=(float**)malloc(N*sizeof(float*));
	for (i = 0; i < N; i++)
	{
		D[i]=(float*)malloc(N*sizeof(float));   
	}
	
	
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			D[i][j] = 0.0;
		}
	}
	
	// Create Dense matrix with random edges
	for (i = 0; i < N; i++)
	{
		D[i][i] = 1;
		for (j = 0; j < N; j++)
		{
			// if edge exists
			if (D[i][j] < 1)
			{
				// Whether edge will exist
				int rand_num = rand() % 2;
				int rand_num_2 = rand() % 2;

				if (rand_num == rand_num_2 && rand_num == 1)
				{
					D[i][j] = 1.0;
					D[j][i] = 1.0;
				}
				else
				{
					D[i][j] = 0.0;
				}
				/*
				D[i][j] = (float) rand_num;
				
				// Edge has to be connected both ways
				if (D[i][j] == 1)
				{
					D[j][i] = 1;
				}*/
			}
		}
	}	
	
	// Initialize color array 
	int* color_arr = (int*)malloc(N*sizeof(int)); 
	for (i = 0; i < N; i++)
	{
		color_arr[i] = 0;
	}
	

	// calculate the sparseness of the matrix
	float sparseness = check_sparseness(D, N);
	
	// start timer for function execution time
	startTime = omp_get_wtime();

	if (sparseness > 0.50)
	{
		printf ("CSR\n");
		
		// Convert Dense matrix into CSR matrix
		create_csr (D, COL, ROW_PTR, N);
		// Color CSR graph
		color_csr_graph (V, ROW_PTR, COL, color_arr, N);
		
		// stop timer
		endTime = omp_get_wtime();
		// output runtime
		printf("Runtime = %.16e\n",endTime-startTime);
		
		// Check whether the graph is correctly colored
		int correct = 1;
		for (i = 0; i < N; i++)
		{
			for (j = ROW_PTR[i]; j < ROW_PTR[i+1]; j++)
			{
				if (i != COL[j] && color_arr[i] == color_arr[COL[j]])
				{
					correct = 0;
				}
			}
		}
		printf ("CORRECT: %d\n", correct);
		
	}
	else
	{
		printf ("Dense\n");
		
		// create the graph with random value
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				D[i][j] = V[j] * D[i][j];
			}
		}

		// Color dense graph
		color_graph (D, color_arr, N);
		
		// stop timer
		endTime = omp_get_wtime();
		// output runtime
		printf("Runtime = %.16e\n",endTime-startTime);
		
		int correct = 1;	
		for (i = 0; i < N; i++)
		{
			int cur_color = color_arr[i];
			for (j = 0; j < N; j++)
			{
				if (i != j && D[i][j] >= 1 && color_arr[j] == cur_color)
				{
					correct = 0;
				}
			}
		}
		printf ("CORRECT: %d\n", correct);
	}
		
	
	return 0;
}