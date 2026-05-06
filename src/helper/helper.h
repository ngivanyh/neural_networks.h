#ifndef HELPER_INCLUDED
#define HELPER_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum NN_RETURN_CODES {
    NN_RETURN_SUCCESS,
    NN_RETURN_ERROR_NULLPTR,
    NN_RETURN_ERROR_INVALID_ARGUMENTS,
    NN_RETURN_ERROR_ALLOC_UNSUCCESSFUL
} NN_RETURN_CODES;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HELPER_INCLUDED