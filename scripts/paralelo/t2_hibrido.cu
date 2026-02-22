/***************************************************************************************
 * * t2_hibrido.cu: Multiplicación de matrices híbrida (MPI + CUDA)
 * Modelo Maestro-Trabajador usando MPI para distribución de tareas
 * y CUDA para el cálculo paralelo masivo en cada nodo trabajador.
 *
 **************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include <mpi.h>
#include <cuda_runtime.h>

/* --- DEFINES --- */
#define SILENT 0
#define VERBOSE 1
#define TRUE 1
#define MASTER 0

#define TAG_SENDTASK 1
#define TAG_FINISH 2
#define TAG_SENDDIM 3
#define TAG_SENDROW 4

/* --- KERNEL CUDA (MBMT: Many Blocks, Many Threads) --- */
__global__ void matrixMultDev_MBMT(float *A, float *B, float *C, int f, int c1, int c2)
{
    int row, column, k;
    float sum;

    row = blockIdx.y * blockDim.y + threadIdx.y;
    column = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < f && column < c2)
    {
        sum = 0.0;
        for(k = 0; k < c1; k = k + 1)
        {
            sum = sum + (A[row * c1 + k] * B[k * c2 + column]);
        }
        C[row * c2 + column] = sum;
    }
}

/* --- FUNCIONES AUXILIARES --- */

void Usage(char *message)
{
    printf("Usage: mpirun -np P %s < datafile.txt\n", message);
    printf("P: cantidad de nodos (1 Maestro, P-1 Trabajadores)\n");
}

// Generación de datos en formato lineal (1D) para compatibilidad con CUDA
void genDataLinear(int f, int c1, int c2, float **A, float **B)
{
    int i;
    *A = (float *) malloc(f * c1 * sizeof(float));
    *B = (float *) malloc(c1 * c2 * sizeof(float));

    for(i = 0; i < (f * c1); i = i + 1) 
    {
        (*A)[i] = 1.0;
    }
    
    for(i = 0; i < (c1 * c2); i = i + 1) 
    {
        (*B)[i] = 2.0;
    }
}

void printMatrixLinear(float *A, int rows, int columns)
{
    int i, j;
    printf("\n");
    for(i = 0; i < rows; i = i + 1)
    {
        for(j = 0; j < columns; j = j + 1)
        {
            printf("%.1f\t", A[i * columns + j]);
        }
        printf("\n");
    }
}

/* --- PROCESO PRINCIPAL --- */

void Process(int mode, int n_node, int n_task)
{
    int nodeID, tasks_sent, workers_active;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);

    if(nodeID == MASTER)
    {
        int dim_Send[3], dim_Recv[3];
        int f_R, c1_R, c2_R;
        float *A_R, *B_R, *C_R;
        int i;

        tasks_sent = 0;
        workers_active = 0;

        // Asignación inicial
        for(i = 1; i < n_node; i = i + 1)
        {
            if(tasks_sent < n_task)
            {
                scanf("%d %d %d", &dim_Send[0], &dim_Send[1], &dim_Send[2]);
                MPI_Send(&dim_Send, 3, MPI_INT, i, TAG_SENDTASK, MPI_COMM_WORLD);
                tasks_sent = tasks_sent + 1;
                workers_active = workers_active + 1;
            }
            else
            {
                MPI_Send(NULL, 0, MPI_INT, i, TAG_FINISH, MPI_COMM_WORLD);
            }
        }

        while(workers_active > 0)
        {
            MPI_Recv(&dim_Recv, 3, MPI_INT, MPI_ANY_SOURCE, TAG_SENDDIM, MPI_COMM_WORLD, &status);
            f_R = dim_Recv[0]; 
            c1_R = dim_Recv[1]; 
            c2_R = dim_Recv[2];

            A_R = (float *) malloc(f_R * c1_R * sizeof(float));
            B_R = (float *) malloc(c1_R * c2_R * sizeof(float));
            C_R = (float *) malloc(f_R * c2_R * sizeof(float));

            // Recepción de datos (manteniendo lógica de t2.c por filas)
            for(i = 0; i < f_R; i = i + 1) 
            {
                MPI_Recv(&A_R[i * c1_R], c1_R, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);
            }
            for(i = 0; i < c1_R; i = i + 1) 
            {
                MPI_Recv(&B_R[i * c2_R], c2_R, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);
            }
            for(i = 0; i < f_R; i = i + 1) 
            {
                MPI_Recv(&C_R[i * c2_R], c2_R, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);
            }

            if(mode == VERBOSE)
            {
                printf("\nMaestro: Matrices recibidas del Trabajador %d\n", status.MPI_SOURCE);
                printMatrixLinear(C_R, f_R, c2_R);
            }

            free(A_R); 
            free(B_R); 
            free(C_R);

            if(tasks_sent < n_task)
            {
                scanf("%d %d %d", &dim_Send[0], &dim_Send[1], &dim_Send[2]);
                MPI_Send(&dim_Send, 3, MPI_INT, status.MPI_SOURCE, TAG_SENDTASK, MPI_COMM_WORLD);
                tasks_sent = tasks_sent + 1;
            }
            else
            {
                MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, TAG_FINISH, MPI_COMM_WORLD);
                workers_active = workers_active - 1;
            }
        }
    }
    else // NODO TRABAJADOR (CUDA)
    {
        int dims[3], f, c1, c2, i;
        float *h_A, *h_B, *h_C;
        float *d_A, *d_B, *d_C;

        while(TRUE)
        {
            MPI_Recv(&dims, 3, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == TAG_FINISH) 
            {
                break;
            }

            f = dims[0]; 
            c1 = dims[1]; 
            c2 = dims[2];
            
            genDataLinear(f, c1, c2, &h_A, &h_B);
            h_C = (float *) malloc(f * c2 * sizeof(float));

            // CUDA: Reservar y Copiar
            cudaMalloc((void **) &d_A, f * c1 * sizeof(float));
            cudaMalloc((void **) &d_B, c1 * c2 * sizeof(float));
            cudaMalloc((void **) &d_C, f * c2 * sizeof(float));

            cudaMemcpy(d_A, h_A, f * c1 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, c1 * c2 * sizeof(float), cudaMemcpyHostToDevice);

            // Configuración del Grid (MBMT)
            dim3 block(32, 32);
            dim3 grid((c2 + block.x - 1) / block.x, (f + block.y - 1) / block.y);

            matrixMultDev_MBMT<<<grid, block>>>(d_A, d_B, d_C, f, c1, c2);
            cudaDeviceSynchronize();

            cudaMemcpy(h_C, d_C, f * c2 * sizeof(float), cudaMemcpyDeviceToHost);

            // Retornar resultados al Maestro
            MPI_Send(dims, 3, MPI_INT, MASTER, TAG_SENDDIM, MPI_COMM_WORLD);
            
            for(i = 0; i < f; i = i + 1) 
            {
                MPI_Send(&h_A[i * c1], c1, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);
            }
            
            for(i = 0; i < c1; i = i + 1) 
            {
                MPI_Send(&h_B[i * c2], c2, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);
            }
            
            for(i = 0; i < f; i = i + 1) 
            {
                MPI_Send(&h_C[i * c2], c2, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);
            }

            // Limpieza
            cudaFree(d_A); 
            cudaFree(d_B); 
            cudaFree(d_C);
            free(h_A); 
            free(h_B); 
            free(h_C);
        }
    }
}

int main(int argc, char **argv)
{
    int n_task, n_nodes, nodeID, mode;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);

    if(argc < 2) 
    {
        if(nodeID == MASTER) 
        {
            Usage(argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if(strcmp(argv[1], "-V") == 0)
    {
        mode = VERBOSE;
    }
    else
    {
        mode = SILENT;
    }

    if(nodeID == MASTER)
    {
        scanf("%d", &n_task);
        printf("Maestro: Procesando %d tareas con %d nodos...\n", n_task, n_nodes);
    }

    Process(mode, n_nodes, n_task);

    MPI_Finalize();
    return 0;
}