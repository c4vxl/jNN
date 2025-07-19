package de.c4vxl.core.optim;

import de.c4vxl.core.optim.type.AbstractOptimizer;
import de.c4vxl.core.tensor.Tensor;

import java.util.List;

public class SGDOptimizer extends AbstractOptimizer {
    public SGDOptimizer(List<Tensor<?>> parameters, double learningRate) {
        super(parameters, learningRate);
    }

    @Override
    public void step() {
        for (Tensor<?> parameter : this.parameters)
            this.handle(parameter);
    }

    private <T> void handle(Tensor<T> parameter) {
        // data = data - learningRate * grad
        Tensor<T> lr = Tensor.filled(parameter.dtype.parse(this.learningRate), parameter.shape.dimensions);
        Tensor<T> optimized = parameter.sub(lr.mul(parameter.grad));

        // Update parameter
        // Detach to prevent gradients from being reused in next training step
        parameter.update(optimized.detach(true));
    }
}