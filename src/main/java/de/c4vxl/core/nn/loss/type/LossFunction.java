package de.c4vxl.core.nn.loss.type;

import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.tensor.Tensor;

public abstract class LossFunction extends Module {
    public abstract <T> Tensor<T> forward(Tensor<T> input, Tensor<T> label);
}
