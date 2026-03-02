/***************************************************************************************
 * * t2_hibrido.cu: Programa que calcula la multiplicación de dos matrices de manera paralela
 * a través del modelo de "granja" o "maestro-trabajador" usando MPI para 
 * distribución de tareas y CUDA (MBMT) para el cálculo paralelo masivo 
 * en cada nodo trabajador.
 *
 * Programmer: Leandro Aballay Henriquez - Delian Santis Lopez
 *
 * Santiago de Chile, 05/01/2026
 *
 **************************************************************************************/
/*
    *** LIBRERÍAS ***
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include <mpi.h>
#include <cuda_runtime.h>

/*
    *** DEFINE's ***
*/

// *** Modos de impresión de resultados ***
#define SILENT 0
#define VERBOSE 1

// *** Constantes lógicas ***
#define TRUE 1

// *** Etiquetas para nodos ***
#define MASTER 0

// *** Etiquetas para tareas *** 
#define TAG_SENDTASK 1  // Etiqueta para mandar tareas a los nodos trabajadores
#define TAG_FINISH 2  // Etiqueta para que el maestro le diga al trabajador que se apague si no tiene más tareas que asignar
#define TAG_SENDDIM 3 // Etiqueta para que el trabajador le mande al maestro las dimensiones de la matriz C
#define TAG_SENDROW 4 // Etiqueta para transferencia de datos (ahora transfiere matrices completas)

/*
    *** VARIABLES GLOBALES ***
*/
int nodeID;
MPI_Status status;

/*
    *** FUNCIONES ***
*/

void Usage(char *message)
{
    printf("Usage: mpirun -np P %s -O < datafile.txt\n\n", message);
    printf("Where: O in {V: Verbose; S: Silent}\n");
    printf("       P: cantidad de nodos\n");
}

// genDataLinear: lee el archivo de entrada y llena las matrices en formato 1D
void genDataLinear(int f, int c1, int c2, float **A, float **B)
{
    int i;
    *A = (float *) malloc(f * c1 * sizeof(float));
    *B = (float *) malloc(c1 * c2 * sizeof(float));

    // Llenar matriz lineal
    for(i = 0; i < (f * c1); i = i + 1) 
    {
        (*A)[i] = 1.0;
    }
    
    // Llenar matriz lineal
    for(i = 0; i < (c1 * c2); i = i + 1) 
    {
        (*B)[i] = 2.0;
    }
}

// printMatrixLinear: imprime en pantalla la matriz en formato 1D
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

// matrixMultDev_MBMT: KERNEL CUDA (Many Blocks, Many Threads)
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

// Process: función que se encarga de establecer la comunicación y tareas entre nodo maestro y nodos trabajadores
void Process(int mode, int n_node, int n_task)
{
    int tasks_sent; // Número de tareas enviadas a los nodos trabajadores
    int workers_active; // Número de trabajadores activos
    
    clock_t CPU_start, CPU_finish;
    time_t Wall_start, Wall_finish;
    float CPU_time;
    long Wall_time;

    int i;

    if(nodeID == MASTER) // Nodo maestro
    {
        int dim_Send[3], dim_Recv[3];
        int f_Recv, c1_Recv, c2_Recv;
        float *A_Recv, *B_Recv, *C_Recv;

        tasks_sent = 0;
        workers_active = 0;

        CPU_start = clock();
        Wall_start = time(NULL);

        /*
            *** Master: Asignación de tareas a nodos trabajadores
        */
        for(i = 1; i < n_node; i = i + 1)
        {
            printf("\nMaestro: Asignando tarea %d\n", i);
            if(tasks_sent < n_task)
            {
                scanf("%d %d %d", &dim_Send[0], &dim_Send[1], &dim_Send[2]);
                
                MPI_Send(&dim_Send, 3, MPI_INT, i, TAG_SENDTASK, MPI_COMM_WORLD); // Envío de las dimensiones a los trabajadores
                tasks_sent = tasks_sent + 1;
                workers_active = workers_active + 1;
            }
            else
            {
                MPI_Send(NULL, 0, MPI_INT, i, TAG_FINISH, MPI_COMM_WORLD); // Si no hay más tareas por asignar, se apagan los nodos
            }
        }

        // En caso de que queden tareas por asignar, se le asigna una a un nodo que se haya desocupado
        while(workers_active > 0)
        {
            MPI_Recv(&dim_Recv, 3, MPI_INT, MPI_ANY_SOURCE, TAG_SENDDIM, MPI_COMM_WORLD, &status); // El maestro recibe las dimensiones
            f_Recv = dim_Recv[0]; 
            c1_Recv = dim_Recv[1]; 
            c2_Recv = dim_Recv[2];

            // Asignación de memoria para matrices del maestro (formato lineal)
            A_Recv = (float *) malloc(f_Recv * c1_Recv * sizeof(float));
            B_Recv = (float *) malloc(c1_Recv * c2_Recv * sizeof(float));
            C_Recv = (float *) malloc(f_Recv * c2_Recv * sizeof(float));

            // Maestro recibe matrices completas del trabajador (Optimizado)
            MPI_Recv(A_Recv, f_Recv * c1_Recv, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);
            MPI_Recv(B_Recv, c1_Recv * c2_Recv, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);
            MPI_Recv(C_Recv, f_Recv * c2_Recv, MPI_FLOAT, status.MPI_SOURCE, TAG_SENDROW, MPI_COMM_WORLD, &status);

            if(mode == VERBOSE)
            {
                printf("\nMaestro: Mostrando matrices del Trabajador %d\n", status.MPI_SOURCE);
                printf("\nMatriz A:\n");
                printMatrixLinear(A_Recv, f_Recv, c1_Recv);
                printf("\nMatriz B:\n");
                printMatrixLinear(B_Recv, c1_Recv, c2_Recv);
                printf("\nMatriz C resultante:\n");
                printMatrixLinear(C_Recv, f_Recv, c2_Recv);
            }

            // Liberación de memoria
            free(A_Recv); 
            free(B_Recv); 
            free(C_Recv);

            if(tasks_sent < n_task)
            {
                scanf("%d %d %d", &dim_Send[0], &dim_Send[1], &dim_Send[2]);
                
                MPI_Send(&dim_Send, 3, MPI_INT, status.MPI_SOURCE, TAG_SENDTASK, MPI_COMM_WORLD); // Se le asigna al mismo nodo que se desocupó
                tasks_sent = tasks_sent + 1;
            }
            else
            {
                MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, TAG_FINISH, MPI_COMM_WORLD); // Si hay nodos sin tareas asignadas, se apagan
                workers_active = workers_active - 1;
            }
        }
        
        Wall_finish = time(NULL);
        CPU_finish = clock();

        CPU_time = (float)((CPU_finish - CPU_start)/CLOCKS_PER_SEC);
        Wall_time = (long)(Wall_finish - Wall_start);

        if(mode == SILENT)
        {
            printf("\n\n-------------------------\n");
            printf("Cantidad de nodos: %d\n", n_node);
            printf("Tiempo de ejecución total (Wall-time): %ld\n", Wall_time);
            printf("-------------------------\n");
        }
    }
    else // Nodo trabajador (CUDA)
    {
        int dim_Send[3], f, c1, c2;
        float *h_A, *h_B, *h_C; // Punteros Host
        float *d_A, *d_B, *d_C; // Punteros Device

        while(TRUE)
        {
            MPI_Recv(&dim_Send, 3, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // El nodo trabajador recibe las dimensiones

            if(status.MPI_TAG == TAG_FINISH) 
            {
                break; // Si el nodo terminó la tarea, se libera
            }
            else if(status.MPI_TAG == TAG_SENDTASK) // Si se le asigna una tarea, realiza el cálculo
            {
                // Asignación de dimensiones
                f = dim_Send[0]; 
                c1 = dim_Send[1]; 
                c2 = dim_Send[2];
                
                genDataLinear(f, c1, c2, &h_A, &h_B);
                h_C = (float *) malloc(f * c2 * sizeof(float));

                /*
                    *** Cálculo de multiplicación de matrices con memoria en GPU (CUDA) ***
                */

                // CUDA: Reservar memoria en Device
                cudaMalloc((void **) &d_A, f * c1 * sizeof(float));
                cudaMalloc((void **) &d_B, c1 * c2 * sizeof(float));
                cudaMalloc((void **) &d_C, f * c2 * sizeof(float));

                // CUDA: Copiar datos de Host a Device
                cudaMemcpy(d_A, h_A, f * c1 * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, h_B, c1 * c2 * sizeof(float), cudaMemcpyHostToDevice);

                // Configuración del Grid (MBMT)
                dim3 block(32, 32);
                dim3 grid((c2 + block.x - 1) / block.x, (f + block.y - 1) / block.y);

                // Ejecución del Kernel
                matrixMultDev_MBMT<<<grid, block>>>(d_A, d_B, d_C, f, c1, c2);
                cudaDeviceSynchronize();

                // CUDA: Copiar resultados de Device a Host
                cudaMemcpy(h_C, d_C, f * c2 * sizeof(float), cudaMemcpyDeviceToHost);

                /*
                    *** Proceso de envío de resultados al maestro ***
                */

                // Envío de las dimensiones al maestro
                MPI_Send(dim_Send, 3, MPI_INT, MASTER, TAG_SENDDIM, MPI_COMM_WORLD);
                
                // Envío de las matrices completas al maestro (Optimizado)
                MPI_Send(h_A, f * c1, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);
                MPI_Send(h_B, c1 * c2, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);
                MPI_Send(h_C, f * c2, MPI_FLOAT, MASTER, TAG_SENDROW, MPI_COMM_WORLD);

                // Liberación de memoria CUDA y Host
                cudaFree(d_A); 
                cudaFree(d_B); 
                cudaFree(d_C);
                free(h_A); 
                free(h_B); 
                free(h_C);
            }
        }
    }
}

int main(int argc, char **argv)
{
    int n_task, n_nodes;
    int me, mode;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    n_task = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);
    MPI_Get_processor_name(processor_name, &me);

    if(argc != 2 || (strcmp(argv[1], "-V") && strcmp(argv[1], "-S")))
    {
        if(nodeID == MASTER) 
        {
            Usage(argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    else
    {
        if(strcmp(argv[1], "-V") == 0)
        {
            mode = VERBOSE;
        }
        else if(strcmp(argv[1], "-S") == 0)
        {
            mode = SILENT;
        }
    }

    if(nodeID == MASTER)
    {
        scanf("%d", &n_task);
        printf("\nMaestro: Se detectaron %d tareas en el archivo\n", n_task);
    }

    Process(mode, n_nodes, n_task);

    MPI_Finalize();

    return 0;
}