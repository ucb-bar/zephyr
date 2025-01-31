/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include <xnnpack.h>  // Include XNNPack headers
#include <math.h>

// Temp: tohost/fromhost symbols for Chipyard usage
typedef struct {
  volatile uint64_t *tohost;
  volatile uint64_t *fromhost;
} HTIF_Type;

volatile uint64_t tohost __attribute__ ((section (".htif")));
volatile uint64_t fromhost __attribute__ ((section (".htif")));

HTIF_Type htif_handler = {
  .tohost = &tohost,
  .fromhost = &fromhost,
};

int main(void) {
    printf("Hello World! Running XNNPACK FP32 Test\n");

    // Initialize XNNPACK
    int status = xnn_initialize(NULL);
    if (status != xnn_status_success) {
        printf("Failed to initialize XNNPack, status code: %d\n", status);
        return -1;
    }
    printf("XNNPACK initialized successfully!\n");

    // Define parameters for a small Fully Connected layer
    const size_t batch_size = 1;
    const size_t input_channels = 4;
    const size_t output_channels = 3;

    // static float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    // static float weights[12] = {
    //     0.1f, 0.2f, 0.3f, 0.4f,
    //     0.5f, 0.6f, 0.7f, 0.8f,
    //     0.9f, 1.0f, 1.1f, 1.2f
    // };
    // static float bias[3] = {0.5f, 0.6f, 0.7f};
    // static float output_data[3];

    static float input_data[4] __attribute__((aligned(16))) = {1.0f, 2.0f, 3.0f, 4.0f};
    static float weights[12] __attribute__((aligned(16))) = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    static float bias[3] __attribute__((aligned(16))) = {0.5f, 0.6f, 0.7f};
    static float output_data[3] __attribute__((aligned(16)));

    // Test malloc
    void* test_alloc = malloc(1024);
    if (!test_alloc) {
        printf("malloc failed!\n");
        return -1;
    }
    printf("malloc succeeded!\n");
    free(test_alloc);

    // Create the Fully Connected operator
    xnn_operator_t fc_op = NULL;
    status = xnn_create_fully_connected_nc_f32(
        input_channels,  // Input size per batch
        output_channels, // Output size per batch
        input_channels,  // Input stride
        output_channels, // Output stride
        weights,         // Weights matrix
        bias,            // Bias vector
        -INFINITY,       // Min activation
        INFINITY,        // Max activation
        0,               // Flags
        NULL,            // Code cache
        NULL,            // Weights cache
        &fc_op);

    if (status != xnn_status_success) {
        printf("Failed to create Fully Connected operator, status code: %d\n", status);
        return -1;
    }

    // Setup the operator
    status = xnn_setup_fully_connected_nc_f32(fc_op, input_data, output_data);
    if (status != xnn_status_success) {
        printf("Failed to setup Fully Connected operator, status code: %d\n", status);
        xnn_delete_operator(fc_op);
        return -1;
    }

    // Run the operator
    status = xnn_run_operator(fc_op, NULL);
    if (status != xnn_status_success) {
        printf("Failed to run Fully Connected operator, status code: %d\n", status);
        xnn_delete_operator(fc_op);
        return -1;
    }

    // Print results
    printf("Fully Connected Output:\n");
    for (size_t i = 0; i < output_channels; i++) {
        printf("%f ", (double)output_data[i]);  // Explicitly cast to avoid warnings
    }
    printf("\n");

    // Cleanup
    xnn_delete_operator(fc_op);
    return 0;
}
