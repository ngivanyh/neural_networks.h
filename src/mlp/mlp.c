#include "mlp.h"
#include <stdlib.h>

NN_RETURN_CODES ResetGrad(MLP * mlp)
{
    if (mlp == NULL) return NN_RETURN_ERROR_NULLPTR;

    double * grad = mlp->grads;
    double * grad_end = grad + mlp->total_grads;
    for (; grad < grad_end; ++grad)
        *grad = 0;

    return NN_RETURN_SUCCESS;
}

NN_RETURN_CODES ForwardPass(double* input_values, size_t inputs, MLP * mlp)
{
    if (mlp == NULL) return NN_RETURN_ERROR_NULLPTR;

    const size_t layer1_neurons = *(mlp->layer_neurons);

    if (inputs != layer1_neurons) return NN_RETURN_ERROR_INVALID_ARGUMENTS;

    // set the static (unchanged) values into activated and values
    for (size_t i = 0; i < layer1_neurons; ++i)
    {
        mlp->values[i] = input_values[i];
        mlp->activated[i] = input_values[i];
    }

    double * cur_val = mlp->values + layer1_neurons;
    double * cur_act = mlp->activated + layer1_neurons;
    double * cur_w = mlp->weights;
    double * cur_b = mlp->biases;
    double * prev_act = mlp->activated;
    size_t * cur_layer = (size_t *) mlp->layer_neurons + 1;

    for (; cur_layer < mlp->layer_neurons + mlp->total_layers; ++cur_layer)
    {
        size_t cur_layer_neurons = *cur_layer;
        size_t prev_layer_neurons = *(cur_layer - 1);
        double * cur_layer_end = cur_val + cur_layer_neurons;

        for (; cur_val < cur_layer_end; ++cur_val, ++cur_act, ++cur_b)
        {
            double weighted_sum = 0.0;

            double * prev_act_end = prev_act + prev_layer_neurons;
            for (; prev_act < prev_act_end; ++prev_act, ++cur_w)
                weighted_sum += *prev_act * *cur_w;
            prev_act -= prev_layer_neurons;
            weighted_sum += *cur_b;

            *cur_val = weighted_sum;
            *cur_act = (weighted_sum < 0) ? 0 : weighted_sum; // ReLU
        }

        prev_act += prev_layer_neurons;
    }

    return NN_RETURN_SUCCESS;
}

MLP * InitializeMLP(size_t total_layers, size_t * layer_neurons)
{
    if (layer_neurons == NULL) return NULL;

    // allocate to a new layer_neurons so we can deinitialize safely every time
    size_t * lns = (size_t *) malloc(total_layers * sizeof(size_t));
    if (lns == NULL) return NULL;

    // calculate neurons and allocate the layer neurons
    size_t total_neurons = 0;
    for (size_t * ln = lns; ln < lns + total_layers; ++ln)
    {
        *ln = layer_neurons[ln - lns];
        total_neurons += *ln;
    }

    // bias calculation
    size_t total_biases = total_neurons - *lns;

    // weight calculation
    size_t total_weights = 0;
    for (size_t * ln = lns + 1; ln < lns + total_layers; ++ln)
        total_weights += *ln * *(ln - 1); // lots and lots of *'s :)

    // grad calculation
    size_t total_grads = total_weights + total_biases + total_neurons;

    // the actual value stores
    double * values = (double *) calloc(total_neurons, sizeof(double));
    if (values == NULL) goto cleanup;

    double * activated = (double *) calloc(total_neurons, sizeof(double));
    if (activated == NULL) goto cleanup;

    double * grads = (double *) calloc(total_grads, sizeof(double));
    if (grads == NULL) goto cleanup;

    double * biases = (double *) calloc(total_biases, sizeof(double));
    if (biases == NULL) goto cleanup;

    double * weights = (double *) calloc(total_weights, sizeof(double));
    if (weights == NULL) goto cleanup;

    MLP * mlp = (MLP *) malloc(sizeof(MLP));
    if (mlp == NULL) return NULL;

    // consts are annoying so we have to use a workaround
    typedef struct {
        size_t total_layers, total_neurons, total_weights, total_biases, total_grads;
        size_t * layer_neurons;
        double * values;
        double * activated;
        double * grads;
        double * biases;
        double * weights;
    } MutableMLP;

    *((MutableMLP *) mlp) = (MutableMLP) {
        .total_layers = total_layers,
        .total_neurons = total_neurons,
        .total_weights = total_weights,
        .total_biases = total_biases,
        .total_grads = total_grads,
        .layer_neurons = lns,
        .values = values,
        .activated = activated,
        .grads = grads,
        .biases = biases,
        .weights = weights,
    };

    return mlp;

cleanup:
    free((void *) lns);
    free((void *) values);
    free((void *) activated);
    free((void *) grads);
    free((void *) biases);
    free((void *) weights);
    return NULL;
}

NN_RETURN_CODES DeinitializeMLP(MLP * mlp)
{
    if (mlp == NULL) return NN_RETURN_ERROR_NULLPTR;

    if (
        mlp->layer_neurons == NULL
        || mlp->values == NULL
        || mlp->activated == NULL
        || mlp->grads == NULL
        || mlp->biases == NULL
        || mlp->weights == NULL
    ) return NN_RETURN_ERROR_NULLPTR;

    free((void *) mlp->layer_neurons);
    free((void *) mlp->values);
    free((void *) mlp->activated);
    free((void *) mlp->grads);
    free((void *) mlp->biases);
    free((void *) mlp->weights);
    free((void *) mlp);

    return NN_RETURN_SUCCESS;
}