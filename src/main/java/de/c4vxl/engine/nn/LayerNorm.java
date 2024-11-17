package de.c4vxl.engine.nn;

import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.module.Module;

/**
 * Applies Layer Normalization over a mini-batch of inputs.
 * <a href="https://arxiv.org/pdf/1607.06450">Read Paper</a>
 *
 * @apiNote EXPERIMENTAL
 * @author c4vxl
 */
public class LayerNorm extends Module {
    public Tensor<Double> weight;
    public Tensor<Double> bias;
    public double epsilon;

    public LayerNorm(int... normalizedShape) { this(1e-5, normalizedShape); }
    public LayerNorm(double epsilon, int... normalizedShape) {
        this.weight = Tensor.ones(Double.class, normalizedShape);
        this.bias = Tensor.zeros(Double.class, normalizedShape);
        this.epsilon = epsilon;
    }

    public <T> Tensor<T> forward(Tensor<T> input) {
        Tensor<Double> doubleTensor = input.asDType(Double.class);

        int batchSize = input.size(0);
        int timeSize = input.size(1);
        int featureSize = input.size(2);

        if (weight.size(0) != featureSize || bias.size(0) != featureSize)
            throw new IllegalArgumentException("Weight and bias must match the feature size.");


        double[][] mean = mean(batchSize, timeSize, featureSize, doubleTensor);
        double[][] variance = variance(batchSize, timeSize, featureSize, mean, doubleTensor);

        // apply normalization, scaling, and shifting
        Tensor<Double> output = new Tensor<>(doubleTensor.dtype, input.shape);
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSize; t++) {
                for (int f = 0; f < featureSize; f++) {
                    double normalized = (doubleTensor.item(b, t, f) - mean[b][t]) / Math.sqrt(variance[b][t] + epsilon);
                    output.set(weight.item(f) * normalized + bias.item(f), b, t, f);
                }
            }
        }

        return output.asDType(input.dtype);
    }

    private double[][] mean(int batchSize, int timeSize, int featureSize, Tensor<Double> doubleTensor) {
        double[][] mean = new double[batchSize][timeSize];

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSize; t++) {
                double sum = 0.0;
                for (int f = 0; f < featureSize; f++) {
                    sum += doubleTensor.item(b, t, f);
                }
                mean[b][t] = sum / featureSize;
            }
        }

        return mean;
    }

    private double[][] variance(int batchSize, int timeSize, int featureSize, double[][] mean, Tensor<Double> doubleTensor) {
        double[][] variance = new double[batchSize][timeSize];

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSize; t++) {
                double sumSq = 0.0;
                for (int f = 0; f < featureSize; f++) {
                    sumSq += Math.pow(doubleTensor.item(b, t, f) - mean[b][t], 2);
                }
                variance[b][t] = sumSq / featureSize;
            }
        }

        return variance;
    }
}