#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iosfwd>
#include <string>
#include <random>
#include <ctime>
#include <chrono>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

using namespace std;
using namespace std::chrono;

const char *kernelSource = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n" \
"__kernel void addition(__global double *a, __global double *b, __global double *c){ \n" \
"  int i = get_global_id(0); \n" \
"  c[i]  = a[i] + b[i];  \n" \
"}";

int main(){
  int size = 10;
  double array_a[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  double array_b[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  // Device input buffers
  cl_mem d_a;
  cl_mem d_b;
  // Device output buffer
  cl_mem d_c;

  cl_platform_id cpPlatform;        // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  cl_program program;               // program
  cl_kernel kernel;                 // kernel

  cl_int error;
  cl_build_status status;
  FILE* programHandle;
  char *programBuffer; char *programLog;
  size_t programSize; size_t logSize;

  // Initialize matrices on host
  double* result_matrix = new double[size];

  size_t bytes_array_a = size * sizeof(double);
  size_t bytes_array_b = size * sizeof(double);
  size_t bytes_result_matrix = size * sizeof(double);

  // Bind to platform
  clGetPlatformIDs(1, &cpPlatform, NULL);

  // Get ID for the device
  clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

  // Create a command queue
  queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, NULL);

  // Build the program executable
  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  if (error != CL_SUCCESS) {
      // check build error and build status first
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS,
              sizeof(cl_build_status), &status, NULL);

      // check build log
      clGetProgramBuildInfo(program, device_id,
              CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
      programLog = (char*) calloc (logSize+1, sizeof(char));
      clGetProgramBuildInfo(program, device_id,
              CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
      printf("Build failed; error=%d, status=%d, programLog:nn%s",
              error, status, programLog);
      free(programLog);
  }


  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "addition", NULL);

  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_array_a, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_array_b, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_result_matrix, NULL, NULL);

  // Write our data set into the input array in device memory
  clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes_array_a, array_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes_array_b, array_b, 0, NULL, NULL);

  // Set the arguments to our compute kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);

  size_t x_range = size;
  size_t global[1] = {x_range};
  // Execute the kernel over the entire range of the data set
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes_result_matrix, result_matrix, 0, NULL, NULL);

  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  for(int i = 0; i<size;i++){
    cout << result_matrix[i] << endl;
  }
}
