#ifndef INCLUDE_MLP
#define INCLUDE_MLP

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdlib.h>

typedef struct {
    const size_t layers; // the amount of layers (input + hidden + output)
    const size_t neurons; // the total amount of neurons everywhere
    const size_t weights; // the total amount of weights
    const size_t biases; // the total amount of biases
    const size_t* layer_neurons; // the amount of neurons in each layer
    // actual value stores
    const double* values;
    const double* grad_values;
    const double* bias_values;
    const double* weight_values;
} MLP;

MLP* MLP_Initialize(size_t layers, size_t neurons, size_t *layer_neurons);
void MLP_Deinitialize(MLP* mlp);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // INCLUDE_MLP

// MLP implementation
#ifdef MLP_IMPLEMENTATION
MLP* MLP_Initialize(size_t layers, size_t neurons, size_t *layer_neurons)
{
    MLP* mlp = malloc(sizeof(MLP));

    // allocate a new layer_neurons so we can always deinitialize safely
    const size_t* lns = malloc(layers * sizeof(int));
    for (size_t* ln = lns; ln < lns + layers; ++ln)
        *ln = layer_neurons[ln - lns];
    mlp->layer_neurons = lns;

    // set constant values
    mlp->layers = layers;
    mlp->neurons = neurons;
    mlp->biases = neurons - *lns;

    size_t w = 0;
    for (size_t* ln, prev_ln = lns + 1, lns; ln < lns + layers; ++ln, ++prev_ln)
        w += *ln * *prev_ln;
    const size_t weights = w;
    mlp->weights = weights;

    // value stores
    mlp->values = calloc(neurons, sizeof(double));
    mlp->grad_values = calloc(weights + mlp->biases, sizeof(double));
    mlp->bias_values = calloc(mlp->biases, sizeof(double));
    mlp->weight_values = calloc(weights, sizeof(double));

    return mlp;
}

void MLP_Deinitialize(MLP *mlp)
{
    free(mlp->layer_neurons);
    free(mlp->values);
    free(mlp->grad_values);
    free(mlp->bias_values);
    free(mlp->weight_values);
    free(mlp);
}

#endif // MLP_IMPLEMENTATION