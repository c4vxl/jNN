package de.c4vxl.engine.activation;

import de.c4vxl.engine.data.Tensor;

import java.util.Arrays;

/**
 * Collection of activation functions
 */
public class Activation {
    /**
     * Rectified linear unit
     */
    public static <T> Tensor<T> ReLU(Tensor<T> input) {
        return input.clone().clip(input.valueOf(0), input.max());
    }

    /**
     * Gaussian Error Linear Unit (GELU)
     */
    public static <T> Tensor<T> GELU(Tensor<T> input) {
        return input.clone().elementWise((a, i) -> {
            double x = ((Number) a).doubleValue();
            double geluValue = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
            return input.valueOf(geluValue);
        });
    }

    /**
     * Compute a softmax on the data in the Tensor
     */
    public static <T> Tensor<T> Softmax(Tensor<T> tensor) {
        // work with Double as dtype here as most of the Math functions only support Double values
        Tensor<Double> doubleTensor = tensor.asDType(Double.class);

        Double[] data = doubleTensor.data;
        double maxLogit = doubleTensor.max();

        double sumExp = 0.0;
        for (double value : data) {
            if (Double.isFinite(value)) sumExp += Math.exp(value - maxLogit);

            // special handling for infinite values
            else if (value == Double.POSITIVE_INFINITY) sumExp += 1.0;
        }

        double finalSumExp = sumExp;
        Double[] softmax = Arrays.stream(data)
                .map(x -> Math.exp(x - maxLogit) / finalSumExp)
                .toArray(Double[]::new);

        return new Tensor<>(softmax, tensor.shape).asDType(tensor.dtype);
    }
}