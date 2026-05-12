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
    float * const values; // non activated weighted sum
    float * const activated; // activated weighted sum
    float * const grads;
    float * const biases;
    float * const weights;
    // end pointers (points to last element + 1)
    const size_t * const layer_neurons_end;
    float * const values_end;
    float * const activated_end;
    float * const grads_end;
    float * const biases_end;
    float * const weights_end;
} MLP;

MLP * InitMLP(size_t total_layers, size_t * layer_neurons);
NN_RETURN_CODES DestroyMLP(MLP * mlp);

NN_RETURN_CODES ForwardPass(float * input_values, size_t inputs, MLP * mlp);
NN_RETURN_CODES BackwardPass(MLP * mlp);
NN_RETURN_CODES ResetGrad(MLP * mlp);

float * GetMLPOutput(MLP * mlp, bool softmax);