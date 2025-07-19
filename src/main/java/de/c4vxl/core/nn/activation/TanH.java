package de.c4vxl.core.nn.activation;

import de.c4vxl.core.tensor.Tensor;

public class TanH extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.tanh(input); }
}
