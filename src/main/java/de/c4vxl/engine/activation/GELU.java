package de.c4vxl.engine.activation;

import de.c4vxl.engine.tensor.Tensor;

public class GELU extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.GELU(input); }
}