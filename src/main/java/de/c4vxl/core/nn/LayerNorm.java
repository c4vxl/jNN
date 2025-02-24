package de.c4vxl.core.nn;

import de.c4vxl.core.module.Module;
import de.c4vxl.core.tensor.Tensor;

/**
 * Applies Layer Normalization over a mini-batch of inputs.
 * <a href="https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html">Read more</a>
 */
public class LayerNorm extends Module {
    public Tensor<Double> weight, bias;
    public double epsilon;

    public LayerNorm(int... normalizedShape) { this(true, normalizedShape); }
    public LayerNorm(boolean bias, int... normalizedShape) { this(1e-05, bias, normalizedShape); }
    public LayerNorm(double eps, boolean bias, int... normalizedShape) {
        this.weight = Tensor.ones(normalizedShape).asDouble();
        this.bias = bias ? Tensor.zeros(normalizedShape).asDouble() : null;
        this.epsilon = eps;
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        Tensor<Double> x = input.asDouble();

        // compute mean and variance
        Tensor<Double> mean = x.mean(-1, true);
        Tensor<Double> variance = x.variance(-1, true);

        // normalize input
        Tensor<Double> xNormalized = x.sub(mean).div(variance.add(epsilon).sqrt());

        // apply scaling
        Tensor<Double> output = xNormalized.mul(weight);

        // apply bias
        if (bias != null)
            output = output.add(bias);

        return output.asDType(input.dtype);
    }
}