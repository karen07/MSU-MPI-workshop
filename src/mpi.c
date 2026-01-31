#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <mpi.h>
#include <omp.h>

int N_x = 127, N_y = 127, N_z = 127, N_T = 20;
double L_x = 1.0, L_y = 1.0, L_z = 1.0, T = 1.0;

double hx, hy, hz, tau;
double h2x, h2y, h2z;
double A, B, C, D;

int num_blocks_x = 1, num_blocks_y = 1, num_blocks_z = 1;

int print_arr_size = 0;
int all_print_z = 0;
int print_arr[10];
int print_arr_num[10];

struct Block {
    int x_rank;
    int y_rank;
    int z_rank;
    int x;
    int y;
    int z;
    int len_x;
    int len_y;
    int len_z;
    int size;
};

struct Block block;

//Solution
double phi(int x, int y, int z, double t)
{
    return sin(A * x) * sin(B * y) * sin(C * z) * cos(D * t * tau);
}

//Initial condition
double phi0(int x, int y, int z)
{
    //return 0;
    return sin(A * x) * sin(B * y) * sin(C * z);
}

//Initial speed condition
double phi0t(int x, int y, int z)
{
    (void)x;
    (void)y;
    (void)z;

    return 0;
}

//Boundary condition
double phi_x_0(int y, int z, double t)
{
    (void)y;
    (void)z;
    (void)t;

    return 0;
}

double phi_x_1(int y, int z, double t)
{
    (void)y;
    (void)z;
    (void)t;

    return 0;
}

double phi_y_0(int x, int z, double t)
{
    (void)x;
    (void)z;
    (void)t;

    return 0;
}

double phi_y_1(int x, int z, double t)
{
    (void)x;
    (void)z;
    (void)t;

    return 0;
}

double phi_z_0(int x, int y, double t)
{
    (void)x;
    (void)y;
    (void)t;

    return 0;
}

double phi_z_1(int x, int y, double t)
{
    (void)x;
    (void)y;
    (void)t;

    return 0;
}

double source(int x, int y, int z, double t)
{
    return 0;
    /*if (x == N_x / 3 && y == N_y / 3 && z == N_z / 3)
        return N_x * N_y * N_z * sin(t * tau * 100);
    else
        return 0;*/
    return N_x * N_y * N_z *
           exp(-((x - N_x / 3) * (x - N_x / 3) + (y - N_y / 3) * (y - N_y / 3) +
                 (z - N_z / 3) * (z - N_z / 3))) *
           sin(t * tau * 100);
}

int get_p(int i, int j, int p)
{
    return i + j * block.len_x + p * block.len_x * block.len_y;
}

double laplace(int x, int y, int z, double *arr)
{
    double q = -2 * arr[get_p(x, y, z)];
    double laplace_x = (arr[get_p(x - 1, y, z)] + q + arr[get_p(x + 1, y, z)]) / h2x;
    double laplace_y = (arr[get_p(x, y - 1, z)] + q + arr[get_p(x, y + 1, z)]) / h2y;
    double laplace_z = (arr[get_p(x, y, z - 1)] + q + arr[get_p(x, y, z + 1)]) / h2z;
    return laplace_x + laplace_y + laplace_z;
}

int from_cord_rank(int x, int y, int z)
{
    if (x == -1)
        x = num_blocks_x - 1;
    if (y == -1)
        y = num_blocks_y - 1;
    if (z == -1)
        z = num_blocks_z - 1;

    if (x == num_blocks_x)
        x = 0;
    if (y == num_blocks_y)
        y = 0;
    if (z == num_blocks_z)
        z = 0;

    return x + y * num_blocks_x + z * num_blocks_x * num_blocks_y;
}

int satmod(int a, int b)
{
    if (a == -1)
        return b - 1;
    if (a == b + 1)
        return 1;
    return a;
}

float ReverseFloat(const float inFloat)
{
    float retVal;
    char *floatToConvert = (char *)&inFloat;
    char *returnFloat = (char *)&retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

void write_to_file_full(double *in, int t, int rank, int size)
{
    int init_seek = 0;
    FILE *fptr = NULL;
    char out_name[100];

    sprintf(out_name, "plot/out_%d.vtk", t);
    if (rank == 0) {
        fptr = fopen(out_name, "wb");
        fprintf(fptr, "# vtk DataFile Version 2.0\n");
        fprintf(fptr, "Wave\n");
        fprintf(fptr, "BINARY\n");
        fprintf(fptr, "DATASET STRUCTURED_POINTS\n");
        fprintf(fptr, "DIMENSIONS %d %d %d\n", N_x + 1, N_y + 1, N_z + 1);
        fprintf(fptr, "ASPECT_RATIO %f %f %f\n", L_x / (N_x + 1), L_y / (N_y + 1), L_z / (N_z + 1));
        fprintf(fptr, "ORIGIN %d %d %d\n", 0, 0, 0);
        fprintf(fptr, "POINT_DATA %d\n", (N_x + 1) * (N_y + 1) * (N_z + 1));
        fprintf(fptr, "SCALARS value float 1\n");
        fprintf(fptr, "LOOKUP_TABLE default\n");
        init_seek = ftell(fptr);
        fclose(fptr);
    }

    MPI_Bcast(&init_seek, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int proc = 0; proc < size; proc++) {
        if (rank == proc) {
            fptr = fopen(out_name, "r+b");
            for (int p = 1; p < block.len_z - 1; p++) {
                for (int j = 1; j < block.len_y - 1; j++) {
                    fseek(fptr,
                          sizeof(float) * (block.x + (N_x + 1) * (block.y + j - 1) +
                                           (N_x + 1) * (N_y + 1) * (block.z + p - 1)) +
                              init_seek,
                          SEEK_SET);
                    for (int i = 1; i < block.len_x - 1; i++) {
                        float tmp1 = ReverseFloat(in[get_p(i, j, p)]);
                        fwrite(&tmp1, sizeof(float), 1, fptr);
                    }
                }
            }
            fclose(fptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void write_to_file(double *in, int t, int rank, int size)
{
    int init_seek = 0;
    FILE *fptr = NULL;
    char out_name[100];

    sprintf(out_name, "plot/out_%d.vtk", t);
    if (rank == 0) {
        fptr = fopen(out_name, "wb");
        fprintf(fptr, "# vtk DataFile Version 2.0\n");
        fprintf(fptr, "Wave\n");
        fprintf(fptr, "BINARY\n");
        fprintf(fptr, "DATASET STRUCTURED_POINTS\n");
        fprintf(fptr, "DIMENSIONS %d %d %d\n", N_x + 1, N_y + 1, all_print_z);
        fprintf(fptr, "ASPECT_RATIO %f %f %f\n", L_x / (N_x + 1), L_y / (N_y + 1),
                L_z / all_print_z);
        fprintf(fptr, "ORIGIN %d %d %d\n", 0, 0, 0);
        fprintf(fptr, "POINT_DATA %d\n", (N_x + 1) * (N_y + 1) * all_print_z);
        fprintf(fptr, "SCALARS value float 1\n");
        fprintf(fptr, "LOOKUP_TABLE default\n");
        init_seek = ftell(fptr);
        fclose(fptr);
    }

    MPI_Bcast(&init_seek, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int proc = 0; proc < size; proc++) {
        if (rank == proc) {
            fptr = fopen(out_name, "r+b");
            for (int p = 0; p < print_arr_size; p++) {
                for (int j = 1; j < block.len_y - 1; j++) {
                    fseek(fptr,
                          sizeof(float) * (block.x + (N_x + 1) * (block.y + j - 1) +
                                           (N_x + 1) * (N_y + 1) * (print_arr_num[p])) +
                              init_seek,
                          SEEK_SET);
                    for (int i = 1; i < block.len_x - 1; i++) {
                        float tmp1 = ReverseFloat(in[get_p(i, j, print_arr[p])]);
                        fwrite(&tmp1, sizeof(float), 1, fptr);
                    }
                }
            }
            fclose(fptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Request request[12];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((size & (size - 1))) {
        if (rank == 0) {
            printf("Bad comm size\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        N_x = N_y = N_z = atoi(argv[1]);
    }

    hx = L_x / N_x, hy = L_y / N_y, hz = L_z / N_z, tau = fmin(hx, fmin(hy, hz)) / 2;
    h2x = hx * hx, h2y = hy * hy, h2z = hz * hz;
    A = 2.0 * M_PI / N_x, B = 2.0 * M_PI / N_y, C = 2.0 * M_PI / N_z,
    D = sqrt(A * A / h2x + B * B / h2y + C * C / h2z);

    double time_start = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        time_start = MPI_Wtime();
        printf("Size:%d  tau:%f  period:%f  h^2+tau^2:%f  processors:%d  OMP:%d\n", N_x + 1, tau,
               2 * M_PI / D / tau, tau * tau + h2x, size, omp_get_max_threads());
    }
    fflush(stdout);

    int tmp = size, md = 0;
    while (tmp != 1) {
        if (md == 0)
            num_blocks_x *= 2;
        else if (md == 1)
            num_blocks_y *= 2;
        else
            num_blocks_z *= 2;
        tmp /= 2;
        md = (md + 1) % 3;
    }

    block.len_x = (N_x + 1) / num_blocks_x + 2;
    block.len_y = (N_y + 1) / num_blocks_y + 2;
    block.len_z = (N_z + 1) / num_blocks_z + 2;

    block.z_rank = rank / (num_blocks_x * num_blocks_y);
    tmp = rank % (num_blocks_x * num_blocks_y);
    block.y_rank = tmp / num_blocks_x;
    block.x_rank = tmp % num_blocks_x;

    block.z = (block.len_z - 2) * block.z_rank;
    block.y = (block.len_y - 2) * block.y_rank;
    block.x = (block.len_x - 2) * block.x_rank;

    block.size = block.len_x * block.len_y * block.len_z;

#ifdef data_write
    for (float z_step = 1.0; z_step < N_z; z_step += (N_z - 2.0) / 3) {
        if ((block.z <= z_step) && (z_step < block.z + block.len_z - 2)) {
            print_arr[print_arr_size] = z_step - block.z + 1;
            print_arr_num[print_arr_size] = all_print_z;
            print_arr_size++;
        }
        all_print_z++;
    }
#endif

    double *arr_1 = (double *)malloc(sizeof(double) * block.size);
    double *arr_2 = (double *)malloc(sizeof(double) * block.size);

#pragma omp parallel for
    for (int p = 0; p < block.len_z; p++) {
        for (int j = 0; j < block.len_y; j++) {
            for (int i = 0; i < block.len_x; i++) {
                arr_1[get_p(i, j, p)] = phi0(satmod(i - 1 + block.x, N_x),
                                             satmod(j - 1 + block.y, N_y),
                                             satmod(p - 1 + block.z, N_z));
            }
        }
    }

    double max_error = 0, error = 0, global_max_error = 0;
    (void)max_error;
    (void)error;
    (void)global_max_error;

#pragma omp parallel for reduction(+ : max_error)
    for (int p = 1; p < block.len_z - 1; p++) {
        for (int j = 1; j < block.len_y - 1; j++) {
            for (int i = 1; i < block.len_x - 1; i++) {
                if (block.x + i - 1 == 0) {
                    arr_2[get_p(i, j, p)] = phi_x_0(block.y + j - 1, block.z + p - 1, 1);
                } else if (block.x + i - 1 == N_x) {
                    arr_2[get_p(i, j, p)] = phi_x_1(block.y + j - 1, block.z + p - 1, 1);
                } else if (block.y + j - 1 == 0) {
                    arr_2[get_p(i, j, p)] = phi_y_0(block.x + i - 1, block.z + p - 1, 1);
                } else if (block.y + j - 1 == N_y) {
                    arr_2[get_p(i, j, p)] = phi_y_1(block.x + i - 1, block.z + p - 1, 1);
                } else if (block.z + p - 1 == 0) {
                    arr_2[get_p(i, j, p)] = phi_z_0(block.x + i - 1, block.y + j - 1, 1);
                } else if (block.z + p - 1 == N_z) {
                    arr_2[get_p(i, j, p)] = phi_z_1(block.x + i - 1, block.y + j - 1, 1);
                } else {
                    arr_2[get_p(i, j, p)] =
                        arr_1[get_p(i, j, p)] +
                        tau * phi0t(block.x + i - 1, block.y + j - 1, block.z + p - 1) +
                        0.5 * tau * tau * laplace(i, j, p, arr_1) +
                        tau * tau * source(block.x + i - 1, block.y + j - 1, block.z + p - 1, 1);
                }
#ifdef error_write
                error = fabs(arr_2[get_p(i, j, p)] -
                             phi(block.x + i - 1, block.y + j - 1, block.z + p - 1, 1));
                max_error += error * error;
#endif
            }
        }
    }

#ifdef error_write
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("L2 error 1:%f  Time:%f\n", sqrt(global_max_error), MPI_Wtime() - time_start);
    }
    fflush(stdout);
#endif

#ifdef data_write
    write_to_file(arr_1, 0, rank, size);
    write_to_file(arr_2, 1, rank, size);
#endif

    // Input buffer
    double *in_buffers[6];
    in_buffers[0] = (double *)malloc(sizeof(double) * block.len_x * block.len_y);
    in_buffers[1] = (double *)malloc(sizeof(double) * block.len_x * block.len_y);
    in_buffers[2] = (double *)malloc(sizeof(double) * block.len_x * block.len_z);
    in_buffers[3] = (double *)malloc(sizeof(double) * block.len_x * block.len_z);
    in_buffers[4] = (double *)malloc(sizeof(double) * block.len_y * block.len_z);
    in_buffers[5] = (double *)malloc(sizeof(double) * block.len_y * block.len_z);

    // Output buffer
    double *out_buffers[6];
    out_buffers[0] = (double *)malloc(sizeof(double) * block.len_x * block.len_y);
    out_buffers[1] = (double *)malloc(sizeof(double) * block.len_x * block.len_y);
    out_buffers[2] = (double *)malloc(sizeof(double) * block.len_x * block.len_z);
    out_buffers[3] = (double *)malloc(sizeof(double) * block.len_x * block.len_z);
    out_buffers[4] = (double *)malloc(sizeof(double) * block.len_y * block.len_z);
    out_buffers[5] = (double *)malloc(sizeof(double) * block.len_y * block.len_z);

    for (int t = 2; t < N_T; t++) {
        //Receive data
        //Down
        MPI_Irecv(in_buffers[0], block.len_x * block.len_y, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank, block.z_rank - 1), 1, MPI_COMM_WORLD,
                  &request[0]);

        //Top
        MPI_Irecv(in_buffers[1], block.len_x * block.len_y, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank, block.z_rank + 1), 2, MPI_COMM_WORLD,
                  &request[1]);

        //Left
        MPI_Irecv(in_buffers[2], block.len_x * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank - 1, block.z_rank), 3, MPI_COMM_WORLD,
                  &request[2]);

        //Right
        MPI_Irecv(in_buffers[3], block.len_x * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank + 1, block.z_rank), 4, MPI_COMM_WORLD,
                  &request[3]);

        //Near
        MPI_Irecv(in_buffers[4], block.len_y * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank - 1, block.y_rank, block.z_rank), 5, MPI_COMM_WORLD,
                  &request[4]);

        //Far
        MPI_Irecv(in_buffers[5], block.len_y * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank + 1, block.y_rank, block.z_rank), 6, MPI_COMM_WORLD,
                  &request[5]);

        //Send data
        //Down
        for (int j = 1; j < block.len_y - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                out_buffers[0][i + j * block.len_x] = arr_2[get_p(i, j, 1)];
        MPI_Isend(out_buffers[0], block.len_y * block.len_x, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank, block.z_rank - 1), 2, MPI_COMM_WORLD,
                  &request[6]);

        //Top
        for (int j = 1; j < block.len_y - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                out_buffers[1][i + j * block.len_x] = arr_2[get_p(i, j, block.len_z - 2)];
        MPI_Isend(out_buffers[1], block.len_y * block.len_x, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank, block.z_rank + 1), 1, MPI_COMM_WORLD,
                  &request[7]);

        //Left
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                out_buffers[2][i + j * block.len_x] = arr_2[get_p(i, 1, j)];
        MPI_Isend(out_buffers[2], block.len_z * block.len_x, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank - 1, block.z_rank), 4, MPI_COMM_WORLD,
                  &request[8]);

        //Right
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                out_buffers[3][i + j * block.len_x] = arr_2[get_p(i, block.len_y - 2, j)];
        MPI_Isend(out_buffers[3], block.len_z * block.len_x, MPI_DOUBLE,
                  from_cord_rank(block.x_rank, block.y_rank + 1, block.z_rank), 3, MPI_COMM_WORLD,
                  &request[9]);

        //Near
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_y - 1; i++)
                out_buffers[4][i + j * block.len_y] = arr_2[get_p(1, i, j)];
        MPI_Isend(out_buffers[4], block.len_y * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank - 1, block.y_rank, block.z_rank), 6, MPI_COMM_WORLD,
                  &request[10]);

        //Far
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_y - 1; i++)
                out_buffers[5][i + j * block.len_y] = arr_2[get_p(block.len_x - 2, i, j)];
        MPI_Isend(out_buffers[5], block.len_y * block.len_z, MPI_DOUBLE,
                  from_cord_rank(block.x_rank + 1, block.y_rank, block.z_rank), 5, MPI_COMM_WORLD,
                  &request[11]);

        //Write data
        //Down
        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_y - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                arr_2[get_p(i, j, 0)] = in_buffers[0][i + j * block.len_x];

        //Top
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_y - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                arr_2[get_p(i, j, block.len_z - 1)] = in_buffers[1][i + j * block.len_x];

        //Left
        MPI_Wait(&request[2], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                arr_2[get_p(i, 0, j)] = in_buffers[2][i + j * block.len_x];

        //Right
        MPI_Wait(&request[3], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_x - 1; i++)
                arr_2[get_p(i, block.len_y - 1, j)] = in_buffers[3][i + j * block.len_x];

        //Near
        MPI_Wait(&request[4], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_y - 1; i++)
                arr_2[get_p(0, i, j)] = in_buffers[4][i + j * block.len_y];

        //Far
        MPI_Wait(&request[5], MPI_STATUS_IGNORE);
        for (int j = 1; j < block.len_z - 1; j++)
            for (int i = 1; i < block.len_y - 1; i++)
                arr_2[get_p(block.len_x - 1, i, j)] = in_buffers[5][i + j * block.len_y];

        for (int i = 6; i < 12; i++) {
            MPI_Wait(&request[i], MPI_STATUS_IGNORE);
        }

        //Solve
        max_error = 0;
#pragma omp parallel for reduction(+ : max_error)
        for (int p = 1; p < block.len_z - 1; p++) {
            for (int j = 1; j < block.len_y - 1; j++) {
                for (int i = 1; i < block.len_x - 1; i++) {
                    if (block.x + i - 1 == 0) {
                        arr_1[get_p(i, j, p)] = phi_x_0(block.y + j - 1, block.z + p - 1, t);
                    } else if (block.x + i - 1 == N_x) {
                        arr_1[get_p(i, j, p)] = phi_x_1(block.y + j - 1, block.z + p - 1, t);
                    } else if (block.y + j - 1 == 0) {
                        arr_1[get_p(i, j, p)] = phi_y_0(block.x + i - 1, block.z + p - 1, t);
                    } else if (block.y + j - 1 == N_y) {
                        arr_1[get_p(i, j, p)] = phi_y_1(block.x + i - 1, block.z + p - 1, t);
                    } else if (block.z + p - 1 == 0) {
                        arr_1[get_p(i, j, p)] = phi_z_0(block.x + i - 1, block.y + j - 1, t);
                    } else if (block.z + p - 1 == N_z) {
                        arr_1[get_p(i, j, p)] = phi_z_1(block.x + i - 1, block.y + j - 1, t);
                    } else {
                        arr_1[get_p(i, j, p)] =
                            2 * arr_2[get_p(i, j, p)] - arr_1[get_p(i, j, p)] +
                            tau * tau * laplace(i, j, p, arr_2) +
                            tau * tau *
                                source(block.x + i - 1, block.y + j - 1, block.z + p - 1, t);
                    }
#ifdef error_write
                    error = fabs(arr_1[get_p(i, j, p)] -
                                 phi(block.x + i - 1, block.y + j - 1, block.z + p - 1, t));
                    max_error += error * error;
#endif
                }
            }
        }

#ifdef error_write
        MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("L2 error %d:%f  Time:%f\n", t, sqrt(global_max_error),
                   MPI_Wtime() - time_start);
        }
        fflush(stdout);
#endif

#ifdef data_write
        write_to_file(arr_1, t, rank, size);
#endif

        double *tmp_point = arr_1;
        arr_1 = arr_2;
        arr_2 = tmp_point;
    }

    for (int i = 0; i < 6; i++) {
        free(in_buffers[i]);
        free(out_buffers[i]);
    }
    free(arr_1);
    free(arr_2);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Spend time:%f\n", MPI_Wtime() - time_start);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
