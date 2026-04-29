#include "mlp.h"

void ResetGrad(MLP * mlp)
{
    if (mlp == NULL)
        return;

    double * grad = mlp->grads;
    for (size_t i = 0; i < mlp->total_grads; ++i)
        *grad++ = 0;
}

void ForwardPass(double* input_values, size_t inputs, MLP * mlp)
{
    if (mlp == NULL)
        return;

    const size_t layer1_neurons = *(mlp->layer_neurons);

    if (inputs != layer1_neurons)
        return;

    // set the static (unchanged) values into activated and values
    for (size_t i = 0; i < layer1_neurons; ++i)
    {
        mlp->values[i] = input_values[i];
        mlp->activated[i] = input_values[i];
    }

    for (
        size_t i = 1, weight_offset = 0, neuron_offset = layer1_neurons;
        i < mlp->total_layers;
        ++i
    )
    {
        size_t cur_layer_neurons = mlp->layer_neurons[i];
        size_t prev_layer_neurons = mlp->layer_neurons[i - 1];

        for (size_t j = 0; j < cur_layer_neurons; ++j, weight_offset += prev_layer_neurons)
        {
            size_t cur_neuron_idx = neuron_offset + j;
            size_t prev_neuron_offset = neuron_offset - prev_layer_neurons;

            double weighted_sum = 0.0;
            for (size_t k = 0; k < prev_layer_neurons; ++k)
                weighted_sum += mlp->activated[prev_neuron_offset + k] * mlp->weights[weight_offset + k];
            weighted_sum += mlp->biases[prev_neuron_offset + j];

            mlp->values[cur_neuron_idx] = weighted_sum;
            mlp->activated[cur_neuron_idx] = (weighted_sum < 0) ? 0.0 : weighted_sum; // ReLU
        }

        neuron_offset += cur_layer_neurons;
    }
}

MLP* InitializeMLP(size_t total_layers, size_t * layer_neurons)
{
    if (layer_neurons == NULL)
        return NULL;

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
    for (size_t* ln = lns + 1; ln < lns + total_layers; ++ln)
        total_weights += *ln * *(ln - 1);

    // grad calculation
    size_t total_grads = total_weights + total_biases + total_neurons;

    // the actual value stores
    double * values = (double *) calloc(total_neurons, sizeof(double));
    double * activated = (double *) calloc(total_neurons, sizeof(double));
    double * biases = (double *) calloc(total_biases, sizeof(double));
    double * weights = (double *) calloc(total_weights, sizeof(double));
    double * grads = (double *) calloc(total_grads, sizeof(double));

    if (values == NULL || activated == NULL || grads == NULL || biases == NULL || weights == NULL)
    {
        free(lns);
        return NULL;
    }

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
}

void DeinitializeMLP(MLP *mlp)
{
    free((void *) mlp->layer_neurons);
    free((void *) mlp->values);
    free((void *) mlp->activated);
    free((void *) mlp->grads);
    free((void *) mlp->biases);
    free((void *) mlp->weights);
    free((void *) mlp);
}