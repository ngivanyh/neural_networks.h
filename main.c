#define MLP_IMPLEMENTATION

#include "mlp.h"
#include <stdio.h>

int main(void)
{
    size_t layer_ns[3] = {2, 2, 1};
    MLP* mlp = MLP_Initialize(3, 5, layer_ns);
    MLP_Deinitialize(mlp);
    return 0;
}