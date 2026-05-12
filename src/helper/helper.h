#ifndef HELPER_INCLUDED
#define HELPER_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum NN_RETURN_CODES {
    NN_RETURN_SUCCESS,
    NN_RETURN_ERROR_NULLPTR,
    NN_RETURN_ERROR_INVALID_ARGUMENTS,
    NN_RETURN_ERROR_ALLOC_UNSUCCESSFUL
} NN_RETURN_CODES;

// float ReLU(float value)
// {
//     return (value < 0) ? 0 : value;
// }

// float Tanh(float value)
// {
//     return (float) tanh((double) value);
// }

// float Sigmoid(float value)
// {
//     return (float) (1.0 / (1.0 + exp(-(double) value)));
// }

// float Softmax(float * values, size_t length, size_t idx)
// {
//     if (values == NULL)
//         return (float) NN_RETURN_ERROR_NULLPTR;

//     float sum = 0.0;
//     for (size_t i = 0; i < length; ++i)
//         sum += exp((double) values[i]);

//     return (float) (exp((double) values[idx]) / sum);
// }

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HELPER_INCLUDED