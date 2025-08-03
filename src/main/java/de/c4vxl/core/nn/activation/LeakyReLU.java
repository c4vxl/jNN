package de.c4vxl.core.nn.activation;

import de.c4vxl.core.nn.activation.type.Activation;
import de.c4vxl.core.nn.activation.type.ActivationFunction;
import de.c4vxl.core.tensor.Tensor;

public class LeakyReLU extends Activation {
    protected double alpha;

    public LeakyReLU() { this(0.01); }
    public LeakyReLU(double alpha) { this.alpha = alpha; }

    @Override public <T> Tensor<T> forward(Tensor<T> input) { return ActivationFunction.LeakyReLU(input, alpha); }
}