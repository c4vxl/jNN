package de.c4vxl.core.optim.type;

import de.c4vxl.core.tensor.Tensor;

import java.util.List;

public abstract class AbstractOptimizer implements Optimizer {
    protected List<Tensor<?>> parameters;
    protected double learningRate;

    public AbstractOptimizer(List<Tensor<?>> parameters, double learningRate) {
        this.parameters = parameters;
        this.learningRate = learningRate;
    }

    /**
     * Clip the gradients at a min and max to prevent explosion towards +/- infinity
     * @param min The minimum value allowed
     * @param max The maximum value allowed
     */
    public void clip_gradients(double min, double max) {
        for (Tensor<?> parameter : this.parameters)
            handle_clip_gradient(parameter, min, max);
    }

    private <T> void handle_clip_gradient(Tensor<T> parameter, double min, double max) {
        parameter.grad = parameter.grad.clip(min, max);
    }

    @Override
    public void zeroGrad() {
        for (Tensor<?> parameter : this.parameters)
            parameter.zeroGrad();
    }
}