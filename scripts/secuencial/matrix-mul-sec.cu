/*
*   matrix-mul-sec.cu: programa que calcula la multiplicación de matrices mediante
*                      paralelización masiva usando CUDA. Este código contiene la
*                      versión secuencial de la asignación de tareas.
*/

/*
    *** LIBRERÍAS ***
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
    *** DEFINE'S ***
*/

#define SILENT 0
#define VERBOSE 1
#define THREADSxBLOCK 1024

#define HOST 0
#define MBOT 1
#define OBMT 2
#define MBMT 3

/*
    *** KERNEL'S ***
*/

// matrixMultHost: multiplicación de matrices ejecutada de forma secuencial en el Host
void matrixMultHost(float *A, float *B, float *C, int f, int c1, int c2)
{
    int i, j, k;

    for(i = 0; i < f; i = i + 1)
    {
        for(j = 0; j < c2; j = j + 1)
        {
            C[i * c2 + j] = 0.0;

            for(k = 0; k < c1; k = k + 1)
            {
                C[i * c2 + j] = (A[i * c1 + k] * B[k * c2 + j]) + C[i * c2 + j];
            }
        }
    }
}

// matrixMultDev_MBOT: kernel que ejecuta la multiplicación en GPU usando un único hilo por cada bloque (MBOT: Many Blocks, One Thread)
__global__ void matrixMultDev_MBOT(float *A, float *B, float *C, int f, int c1, int c2)
{
    int row, column, i;
    float sum;

    row = blockIdx.y;
    column = blockIdx.x;

    if(row < f && column < c2)
    {
        sum = 0.0;

        for(i = 0; i < c1; i = i + 1)
        {
            sum = sum + (A[row * c1 + i] * B[i * c2 + column]);
        }

        C[row * c2 + column] = sum;
    }
}

// matrixMultDev_OBMT: kernel que ejecuta la multiplicación en GPU usando varios hilos en un único bloque (OBMT: One Block, Many Threads)
// Limitación: f * c2 <= 1024 (máximo de hilos por bloque)
__global__ void matrixMultDev_OBMT(float *A, float *B, float *C, int f, int c1, int c2)
{
    int row, column, i;
    float sum;

    row = threadIdx.y;
    column = threadIdx.x;

    if(row < f && column < c2)
    {
        sum = 0.0;

        for(i = 0; i < c1; i = i + 1)
        {
            sum = sum + (A[row * c1 + i] * B[i * c2 + column]);
        }

        C[row * c2 + column] = sum;
    }
}

// matrixMultDev_MBMT: kernel que ejecuta la multiplicación en GPU usando varios hilos en varios bloques (MBMT: Many Blocks, Many Threads)
__global__ void matrixMultDev_MBMT(float *A, float *B, float *C, int f, int c1, int c2)
{
    int row, column, i;
    float sum;

    row = blockIdx.y * blockDim.y + threadIdx.y;
    column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < f && column < c2)
    {
        sum = 0.0;

        for(i = 0; i < c1; i = i + 1)
        {
            sum = sum + (A[row * c1 + i] * B[i * c2 + column]);
        }

        C[row * c2 + column] = sum;
    }
}

/*
    *** FUNCIONES ***
*/

void Usage(char *message)
{
    printf("Usage: %s k -O < datafile.txt\n\n", message);
    printf("Where: O in {V: Verbose; S: Silent}\n");
    printf("       k: modo de paralelización (0: Secuencial, 1: MBOT, 2: OBMT, 3: MBMT)\n");
}

void genData(float **A, float **B, int f, int c1, int c2)
{
    int i;
    int size_A, size_B;

    size_A = f * c1;
    size_B = c1 * c2;

    *A = (float *) malloc(size_A * sizeof(float));
    for(i = 0; i < size_A; i = i + 1)
    {
        (*A)[i] = 1.0;
    }

    *B = (float *) malloc(size_B * sizeof(float));
    for(i = 0; i < size_B; i = i + 1)
    {
        (*B)[i] = 2.0;
    }
}

// printMatrix: Imprime la matriz en pantalla
void printMatrix(float *A, int rows, int columns)
{
    int i, j;

    for(i = 0; i < rows; i = i + 1)
    {
        for(j = 0; j < columns; j = j + 1)
        {
            printf("%.1f\t", A[i * columns + j]);
        }
        printf("\n");
    }
}


// printData: Imprime la información en pantalla según el modo de impresión
void printData(float *A, float *B, float *C, int f, int c1, int c2, int mode)
{
    if(mode == VERBOSE)
    {
        printf("\n--------------------------\n");
        printf("Matrix A:\n");
        printMatrix(A, f, c1);
        printf("\nMatrix B:\n");
        printMatrix(B, c1, c2);
        printf("\nMatrix C:\n");
        printMatrix(C, f, c2);
        printf("\n--------------------------\n");
    }
}


int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C; // Matrices A, B y C de la CPU
    float *d_A, *d_B, *d_C; // Matrices A, B y C de la GPU

    int f, c1, c2; // Dimensiones de las matrices
    unsigned int size_A, size_B, size_C; // Tamaño total (en bytes) de las matrices A, B y C

    int printMode, paralMode; // Modo de impresión en pantalla y modo de paralelización, respectivamente
    int n_task; // Cantidad de tareas a realizar;

    dim3 block, grid;

    int i;

    // Variables para cálculo de tiempo
    clock_t CPU_start, CPU_finish;
    time_t Wall_start, Wall_finish;
    float CPU_time;
    long Wall_time;

    if(argc != 3 || (strcmp(argv[2], "-V") && strcmp(argv[2], "-S")))
    {
        Usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    else
    {
        if(strcmp(argv[2], "-V") == 0)
        {
            printMode = VERBOSE;
        }
        else
        {
            printMode = SILENT;
        }

        paralMode = atoi(argv[1]);

        scanf("%d", &n_task); // Lee la cantidad de tareas a realizar
        printf("Cantidad de tareas a asignar: %d\n\n", n_task);        

        if(paralMode == HOST) // Ejecución de la versión secuencial de la multiplicación de matrices
        {
            CPU_start = clock();
            Wall_start = time(NULL);

            for(i = 0; i < n_task; i = i + 1)
            {
                scanf("%d %d %d", &f, &c1, &c2);  // Lectura de dimesiones
                genData(&h_A, &h_B, f, c1, c2);  // Generación de matrices

                // Asignación de memoria a matriz C de CPU
                h_C = (float *) malloc((f * c2) * sizeof(float));
                
                matrixMultHost(h_A, h_B, h_C, f, c1, c2);

                printData(h_A, h_B, h_C, f, c1, c2, printMode);

                // Liberación de memoria
                free(h_A);
                free(h_B);
                free(h_C);
            }

            Wall_finish = time(NULL);
            CPU_finish = clock();
        }
        else // Ejecución de versión pararlela usando CUDA
        {
            CPU_start = clock();
            Wall_start = time(NULL);

            for(i = 0; i < n_task; i = i + 1)
            {
                scanf("%d %d %d", &f, &c1, &c2);  // Lectura de dimesiones
                genData(&h_A, &h_B, f, c1, c2);  // Generación de matrices

                size_A = f * c1 * sizeof(float);
                size_B = c1 * c2 * sizeof(float);
                size_C = f * c2 * sizeof(float);

                // Asignación de memoria a matriz C de CPU
                h_C = (float *) malloc(size_C);

                // Asignación de memoria en la GPU
                cudaMalloc((void **) &d_A, size_A);
                cudaMalloc((void **) &d_B, size_B);
                cudaMalloc((void **) &d_C, size_C);

                // Copia de memoria desde CPU a GPU
                cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

                if(paralMode == MBOT) // Ejecución modo MBOT
                {
                    block = dim3(1,1);
                    grid = dim3(c2,f);
                    matrixMultDev_MBOT<<< grid, block >>>(d_A, d_B, d_C, f, c1, c2);
                }
                else if(paralMode == OBMT) // Ejecución modo OBMT
                {
                    block = dim3(c2,f);
                    grid = dim3(1,1);
                    matrixMultDev_OBMT<<< grid, block >>>(d_A, d_B, d_C, f, c1, c2);
                }
                else if(paralMode == MBMT) // Ejecución modo MBMT
                {
                    block = dim3(32, 32);
                    grid = dim3((c2 + block.x - 1) / block.x, (f + block.y - 1) / block.y);
                    matrixMultDev_MBMT<<< grid, block >>>(d_A, d_B, d_C, f, c1, c2);
                }

                // Sincronización entre CPU y GPU
                cudaDeviceSynchronize();

                // Copia de memoria desde GPU a CPU
                cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

                printData(h_A, h_B, h_C, f, c1, c2, printMode);

                // Liberación de memoria de matrices en GPU
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);

                // Liberación de memoria de matrices en CPU
                free(h_A);
                free(h_B);
                free(h_C);
            }

            Wall_finish = time(NULL);
            CPU_finish = clock();
        }

        CPU_time = (float)((CPU_finish - CPU_start)/CLOCKS_PER_SEC);
        Wall_time = (double)(Wall_finish - Wall_start);

        if(printMode == SILENT)
        {
            printf("\n--------------------------\n");
            printf("Cantidad de nodos: (agregar a parámetros de la función cuando se implemente con MPICH)");
            printf("\n--------------------------\n");
        }

        printf("\n--------------------------\n");
        printf("Tiempo de CPU (CPU-time): %f\n", CPU_time);
        printf("Tiempo de ejeución total (Wall-time): %ld", Wall_time);
        printf("\n--------------------------\n");
    }

    
    return 0;
}