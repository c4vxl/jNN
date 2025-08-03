package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;

public class ReLUOperation<T> extends ClipOperation<T> {
    public ReLUOperation(Tensor<T> a) {
        super(a, 0, Double.MAX_VALUE);
    }
}