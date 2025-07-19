package de.c4vxl.core.nn.loss;

import de.c4vxl.core.nn.loss.type.LossFunction;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.MSEOperation;

/**
 * This is an implementation of Mean-square-error (MSE for short)
 */
public class MSELoss extends LossFunction {
    @Override
    public <T> Tensor<T> forward(Tensor<T> output, Tensor<T> target) {
        return new MSEOperation<>(output, target).forward();
    }
}