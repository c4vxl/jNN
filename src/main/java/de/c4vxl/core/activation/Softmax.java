package de.c4vxl.core.activation;

import de.c4vxl.core.tensor.Tensor;

public class Softmax extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.Softmax(input); }
}
