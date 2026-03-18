#ifndef INCLUDE_MLP
#define INCLUDE_MLP // prevent repeated includes

#include <signal.h>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdlib.h>

typedef struct {
    const size_t total_layers; // the amount of layers (input + hidden + output)
    const size_t total_neurons; // the total amount of neurons everywhere
    const size_t total_weights; // the total amount of weights
    const size_t total_biases; // the total amount of biases
    const size_t total_grads; // the total amount of gradients
    const size_t* layer_neurons; // the amount of neurons in each layer
    // actual value stores
    const double* values; // non activated weighted sum
    const double* activated; // activated weighted sum
    const double* grads;
    const double* biases;
    const double* weights;
} MLP;

MLP* InitializeMLP(size_t total_layers, size_t *layer_neurons);
void DeinitializeMLP(MLP* mlp);

void ForwardPass(double* inputs, MLP* mlp);
void BackwardPass(MLP* mlp);
void ResetGrad(MLP* mlp);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // INCLUDE_MLP

// MLP implementation
#ifdef MLP_IMPLEMENTATION
void ResetGrad(MLP* mlp)
{
    for (size_t i = 0; i < mlp->total_grads; ++i)
        mlp->grads[i] = 0;
}

void ForwardPass(double* inputs, MLP* mlp)
{
    for (size_t i = 0; i < *(mlp->layer_neurons); ++i)
    {
        mlp->values[i] = inputs[i];
        mlp->activated[i] = inputs[i];
    }

    for (size_t i = 1; i < mlp->total_layers - 1; ++i)
    {
        //
    }
}

void BackwardPass(MLP* mlp)
{

}

MLP* InitializeMLP(size_t total_layers, size_t *layer_neurons)
{
    // allocate to a new layer_neurons so we can deinitialize safely every time
    size_t* lns = malloc(total_layers * sizeof(size_t));
    if (lns == NULL) return NULL;
    // calculate neurons and allocate the layer neurons
    size_t total_neurons = 0;
    for (size_t* ln = lns; ln < lns + total_layers; ++ln)
    {
        *ln = layer_neurons[ln - lns];
        total_neurons += *ln;
    }

    // bias calculation
    size_t total_biases = total_neurons - *lns;

    // weight calculation
    size_t total_weights = 0;
    for (size_t* ln = lns + 1; ln < lns + total_layers; ++ln)
        total_weights += *ln * *(ln - 1);

    // grad calculation
    size_t total_grads = total_weights + total_biases + total_neurons;

    // the actual value stores
    double* values = calloc(total_neurons, sizeof(double));
    double* activated = calloc(total_neurons, sizeof(double));
    double* biases = calloc(total_biases, sizeof(double));
    double* weights = calloc(total_weights, sizeof(double));
    double* grads = calloc(total_grads, sizeof(double));

    if (values == NULL || activated == NULL || grads == NULL || biases == NULL || weights == NULL)
    {
        free(lns);
        return NULL;
    }

    MLP* mlp = malloc(sizeof(MLP));
    if (mlp == NULL) return NULL;

    // consts are annoying so we have to use a workaround
    typedef struct {
        size_t total_layers, total_neurons, total_weights, total_biases, total_grads;
        size_t* layer_neurons;
        double* values; double* activated; double* grads; double* biases; double* weights;
    } MutableMLP;

    *((MutableMLP*) mlp) = (MutableMLP) {
        .total_layers = total_layers,
        .total_neurons = total_neurons,
        .total_biases = total_biases,
        .total_weights = total_weights,
        .total_grads = total_grads,
        .layer_neurons = lns,
        .values = values,
        .activated = activated,
        .grads = grads,
        .biases = biases,
        .weights = weights,
    };

    return mlp;
}

void DeinitializeMLP(MLP *mlp)
{
    free((void*) mlp->layer_neurons);
    free((void*) mlp->values);
    free((void*) mlp->activated);
    free((void*) mlp->grads);
    free((void*) mlp->biases);
    free((void*) mlp->weights);
    free((void*) mlp);
}
#endif // MLP_IMPLEMENTATION