package de.c4vxl.engine.activation;

import de.c4vxl.engine.data.Tensor;

public class Sigmoid extends ActivationFunction {
    @Override
    public <T> Tensor<T> forward(Tensor<T> input) {
        return Activation.Sigmoid(input);
    }
}
