package de.c4vxl.engine.activation;

import de.c4vxl.engine.module.Module;
import de.c4vxl.engine.data.Tensor;

public abstract class ActivationFunction extends Module {
    public abstract <T> Tensor<T> forward(Tensor<T> input);
}