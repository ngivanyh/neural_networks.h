#define MLP_IMPLEMENTATION

#include "mlp.h"
#include <stdio.h>

int main(void)
{
    size_t layer_ns[3] = {2, 2, 1};
    printf("hello\n");
    MLP* mlp = InitializeMLP(3, layer_ns);
    DeinitializeMLP(mlp);
    return 0;
}