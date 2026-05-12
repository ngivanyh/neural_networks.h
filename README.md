# `neural_networks.h`
(Soon to be) A collection of single header libraries for creating, training, and running popular types of neural networks on CPU, written completely in C.

The recommended way to use this library is to use it with the functions and types it provides, the functions in the library will not accomdate for any unintended behavior that may occur should the user differ from what is intended.

## Links to API References
- [`mlp.h`](docs/MLP.md)
- [`bigram.h`](docs/Bigram.md)
- [`rnn.h`](docs/RNN.md)
- [`transformer.h`](docs/Transformer.md)

## Roadmap
The current agenda is to complete `mlp.h` first and refine it until it can do basic to intermediate levels of inference and training capabilites. Then in order `bigram.h`, `rnn.h`, and finally `transformer.h`.

## Helper API Reference

### `NN_RETURN_CODES`

```c
typedef enum NN_RETURN_CODES {
    NN_RETURN_SUCCESS,
    NN_RETURN_ERROR_NULLPTR,
    NN_RETURN_ERROR_INVALID_ARGUMENTS,
    NN_RETURN_ERROR_ALLOC_UNSUCCESSFUL
} NN_RETURN_CODES;
```

An `enum` of return codes for all library functions.

### Activation Functions
`float` receiving and returning activation functions.

#### Supported Activation Functions
- `ReLU`
- `Tanh`
- `Sigmoid`
- `Softmax` (Receives `float *`, `size_t` length, and an index; returns the softmax value at that index)