#include "mlp.h"
#include <stdlib.h>

NN_RETURN_CODES ForwardPass(float* input_values, size_t inputs, MLP * mlp)
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

    float * cur_val = mlp->values + layer1_neurons;
    float * cur_act = mlp->activated + layer1_neurons;
    float * cur_w = mlp->weights;
    float * cur_b = mlp->biases;
    float * prev_act = mlp->activated;
    size_t * cur_layer = (size_t *) mlp->layer_neurons + 1;

    for (; cur_layer < mlp->layer_neurons_end; ++cur_layer)
    {
        size_t cur_layer_neurons = *cur_layer;
        size_t prev_layer_neurons = *(cur_layer - 1);
        float * cur_layer_end = cur_val + cur_layer_neurons;

        for (; cur_val < cur_layer_end; ++cur_val, ++cur_act, ++cur_b)
        {
            float weighted_sum = 0.0;

            float * prev_act_end = prev_act + prev_layer_neurons;
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

float * GetMLPOutput(MLP * mlp, bool softmax)
{
    float a = 0.0;
    return &a;
}

NN_RETURN_CODES ResetGrad(MLP * mlp)
{
    if (mlp == NULL) return NN_RETURN_ERROR_NULLPTR;

    for (float * grad = mlp->grads; grad < mlp->grads_end; ++grad)
        *grad = 0;

    return NN_RETURN_SUCCESS;
}

MLP * InitMLP(size_t total_layers, size_t * layer_neurons)
{
    if (layer_neurons == NULL) return NULL;

    // allocate to a new layer_neurons so we can deinitialize safely every time
    size_t * lns = (size_t *) malloc(total_layers * sizeof(size_t));
    if (lns == NULL) goto cleanup;

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
    float * values = (float *) calloc(total_neurons, sizeof(float));
    if (values == NULL) goto cleanup;

    float * activated = (float *) calloc(total_neurons, sizeof(float));
    if (activated == NULL) goto cleanup;

    float * grads = (float *) calloc(total_grads, sizeof(float));
    if (grads == NULL) goto cleanup;

    float * biases = (float *) calloc(total_biases, sizeof(float));
    if (biases == NULL) goto cleanup;

    float * weights = (float *) calloc(total_weights, sizeof(float));
    if (weights == NULL) goto cleanup;

    MLP * mlp = (MLP *) malloc(sizeof(MLP));
    if (mlp == NULL) return NULL;

    // consts are annoying so we have to use a workaround
    typedef struct {
        size_t total_layers, total_neurons, total_weights, total_biases, total_grads;
        size_t * layer_neurons;
        float * values;
        float * activated;
        float * grads;
        float * biases;
        float * weights;
        size_t * layer_neurons_end;
        float * values_end;
        float * activated_end;
        float * grads_end;
        float * biases_end;
        float * weights_end;
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
        .layer_neurons_end = lns + total_layers,
        .values_end = values + total_neurons,
        .activated_end = activated + total_neurons,
        .grads_end = grads + total_grads,
        .biases_end = biases + total_biases,
        .weights_end = weights + total_weights,
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

NN_RETURN_CODES DestroyMLP(MLP * mlp)
{
    free((void *) mlp->layer_neurons);
    free((void *) mlp->values);
    free((void *) mlp->activated);
    free((void *) mlp->grads);
    free((void *) mlp->biases);
    free((void *) mlp->weights);
    free((void *) mlp);

    return NN_RETURN_SUCCESS;
}