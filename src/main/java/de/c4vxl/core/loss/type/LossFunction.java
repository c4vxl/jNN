package de.c4vxl.core.loss.type;

import de.c4vxl.core.module.Module;
import de.c4vxl.core.tensor.Tensor;

public abstract class LossFunction extends Module {
    public abstract <T> Tensor<T> forward(Tensor<T> input, Tensor<T> label);
}
