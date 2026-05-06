# `mlp.h` API Reference
## Table of Contents
Empty for now.

## `MLP` Type
The core type that stores everything an MLP[^1] needs to operate. All fields are `const`, preventing the tampering of the MLP to reduce potential edge cases. Below is the definition:

```c
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
```

## `InitializeMLP()`
**Parameters: `size_t total_layers, size_t *layer_neurons`**

Initializes a `MLP` struct with the space allocated for weights, biases, activations, and weighted sums.

## `DeinitializeMLP()`
**Parameters: `MLP* mlp`**

Frees the memory allocated when initializing the `MLP`. Meant to be paired with `InitializeMLP()` for safe `MLP` allocation/deallocation.

## `ForwardPass()`
**Parameters: `double* input_values, size_t inputs, MLP* mlp`**

Does a full forward pass on the model.

## `ResetGrad()`
**Parameters: `MLP* mlp`**

Resets all the gradients to 0.

[^1]: The current MLP implementation uses ReLU as its activation function