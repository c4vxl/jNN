package de.c4vxl.core.nn.loss;

import de.c4vxl.core.nn.loss.type.LossFunction;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.CrossEntropyLossOperation;

/**
 * This is an implementation of CrossEntropyLoss
 */
public class CrossEntropyLoss extends LossFunction {
    @Override
    public <T> Tensor<T> forward(Tensor<T> output, Tensor<T> target) {
        return new CrossEntropyLossOperation<>(output, target).forward();
    }
}