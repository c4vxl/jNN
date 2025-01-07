package de.c4vxl.engine.activation;

import de.c4vxl.engine.tensor.Tensor;

public class Softmax extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.Softmax(input); }
}
