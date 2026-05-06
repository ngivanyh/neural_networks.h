#include <stdlib.h>
#include "../helper/helper.h"

// The MLP type
typedef struct {
    const size_t total_layers; // the amount of layers (input + hidden + output)
    const size_t total_neurons; // the total amount of neurons everywhere
    const size_t total_weights; // the total amount of weights
    const size_t total_biases; // the total amount of biases
    const size_t total_grads; // the total amount of gradients
    const size_t * const layer_neurons; // the amount of neurons in each layer
    // actual value stores
    double * const values; // non activated weighted sum
    double * const activated; // activated weighted sum
    double * const grads;
    double * const biases;
    double * const weights;
} MLP;

// typedef union MLP_INIT_RETURNS {
//     int return_code;
//     MLP * mlp;
// } MLP_INIT_RETURNS;

MLP * InitializeMLP(size_t total_layers, size_t * layer_neurons);
NN_RETURN_CODES DeinitializeMLP(MLP * mlp);

NN_RETURN_CODES ForwardPass(double * input_values, size_t inputs, MLP * mlp);
void BackwardPass(MLP * mlp);
NN_RETURN_CODES ResetGrad(MLP * mlp);