package de.c4vxl.engine.activation;

import de.c4vxl.engine.tensor.Tensor;

public class ReLU extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.ReLU(input); }
}
