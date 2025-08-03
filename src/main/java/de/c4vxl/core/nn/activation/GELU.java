package de.c4vxl.core.nn.activation;

import de.c4vxl.core.nn.activation.type.Activation;
import de.c4vxl.core.nn.activation.type.ActivationFunction;
import de.c4vxl.core.tensor.Tensor;

public class GELU extends Activation {
    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.GELU(input); }
}