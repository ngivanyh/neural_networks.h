#include "mlp/mlp.h"
#include <stdio.h>

int main(void)
{
    size_t layer_ns[3] = {2, 2, 1};
    MLP* mlp = InitializeMLP(3, layer_ns);
    if (!mlp) return 1;

    // Manually set weights and biases for verification
    // Layer 1 (2 neurons, 4 weights)
    mlp->weights[0] = 0.5;  mlp->weights[1] = -0.5; // Neuron 1
    mlp->weights[2] = 0.1;  mlp->weights[3] = 0.2;  // Neuron 2
    mlp->biases[0] = 0.1;   mlp->biases[1] = 0.1;

    // Layer 2 (1 neuron, 2 weights)
    mlp->weights[4] = 0.7;  mlp->weights[5] = -0.3;
    mlp->biases[2] = -0.2;

    double inputs[2] = {1.0, -1.0};
    ForwardPass(inputs, 2, mlp);

    printf("Input: [%.1f, %.1f]\n", inputs[0], inputs[1]);
    printf("Output: %lf (Expected: 0.570000)\n", mlp->activated[mlp->total_neurons - 1]);

    DeinitializeMLP(mlp);
    return 0;
}