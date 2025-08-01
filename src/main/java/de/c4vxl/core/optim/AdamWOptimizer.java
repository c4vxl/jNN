package de.c4vxl.core.optim;

import de.c4vxl.core.tensor.Tensor;

import java.util.List;

/**
 * This is the implementation of an AdamW optimizer
 */
public class AdamWOptimizer extends AdamOptimizer {
    private final double weight_decay;

    public AdamWOptimizer(List<Tensor<?>> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0.01);
    }

    public AdamWOptimizer(List<Tensor<?>> parameters, double learningRate, double beta1, double beta2, double epsilon, double weight_decay) {
        super(parameters, learningRate, beta1, beta2, epsilon);
        this.weight_decay = weight_decay;
    }

    @Override
    protected <T> void handle(Tensor<T> parameter) {
        // Apply weight decay
        Tensor<T> decayTerm = parameter.mul(parameter.dtype.parse(this.learningRate * this.weight_decay));

        Tensor<T> updated = parameter.sub(calculateStep(parameter)).sub(decayTerm);

        // Update parameter
        // Detach to prevent gradients from being reused in next training step
        parameter.update(updated.detach(true));
    }
}