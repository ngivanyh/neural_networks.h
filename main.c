#define MLP_IMPLEMENTATION

#include "mlp.h"
#include <stdio.h>

int main(void)
{
    size_t layer_ns[3] = {2, 2, 1};
    MLP* mlp = InitializeMLP(3, layer_ns);
    double inputs[2] = {1.0, -1.0};
    ForwardPass(inputs, 2, mlp);
    printf("%lf\n", mlp->activated[mlp->total_neurons - 1]);
    DeinitializeMLP(mlp);
    return 0;
}