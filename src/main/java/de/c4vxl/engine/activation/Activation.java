package de.c4vxl.engine.activation;

import de.c4vxl.engine.data.Broadcasting;
import de.c4vxl.engine.data.DType;
import de.c4vxl.engine.data.Tensor;
import de.c4vxl.engine.data.TensorUtils;

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
     * Apply Softmax. Dimension = -1 will be used!
     * @param tensor The Tensor to apply the Softmax onto
     */
    public static <T> Tensor<T> Softmax(Tensor<T> tensor) { return Softmax(tensor, -1); }

    /**
     * Apply Softmax over a specified Dimension
     * @param tensor The Tensor to apply the Softmax onto
     * @param dim The dimension to use
     */
    public static <T> Tensor<T> Softmax(Tensor<T> tensor, int dim) {
        // allow negative indexing
        dim = TensorUtils.handleNegativeDim(tensor.shape, dim);

        if (dim < 0 || dim >= tensor.rank())
            throw new IllegalArgumentException("Invalid dimension: " + dim);

        Tensor<Double> doubleTensor = tensor.asDType(Double.class); // use double for precision

        Tensor<Double> result = Tensor.empty().asDType(Double.class).reshapeUnsafe(tensor.shape);

        applySoftmaxRecursive(doubleTensor, result, dim, 0, new int[tensor.rank()], tensor.shape);

        return result
                .asDType(tensor.dtype); // convert back to original dtype
    }

    /**
     * apply softmax on all the data in the Tensor ignoring dimensions
     * @param tensor The Tensor to work with
     */
    public static Tensor<Double> Softmax1d(Tensor<Double> tensor) {
        Double[] data = tensor.data;
        double maxLogit = tensor.max();

        double sumExp = 0.0;
        boolean hasFiniteValue = false;

        for (double value : data) {
            if (Double.isFinite(value)) {
                sumExp += Math.exp(value - maxLogit);
                hasFiniteValue = true;
            }

            // special handling for infinite values
            else if (value == Double.POSITIVE_INFINITY)
                sumExp += 1.0; // exp(POSITIVE_INFINITY - maxLogit) is ~ 1
        }

        if (!hasFiniteValue) {
            Double[] softmax = Arrays.stream(data)
                    .map(x -> x == Double.NEGATIVE_INFINITY ? 1.0 / data.length : 0.0)
                    .toArray(Double[]::new);
            return new Tensor<>(softmax, tensor.shape).asDType(tensor.dtype);
        }


        double finalSumExp = sumExp;
        Double[] softmax = Arrays.stream(data)
                .map(x -> Double.isFinite(x) ? Math.exp(x - maxLogit) / finalSumExp : 0.0)
                .toArray(Double[]::new);

        return new Tensor<>(softmax, tensor.shape).asDType(tensor.dtype);
    }

    // helper method for softmax
    private static void applySoftmaxRecursive(Tensor<Double> source, Tensor<Double> result, int dim, int currentDim, int[] indices, int[] shape) {
        if (currentDim == dim) {
            int targetSize = shape[dim];
            Double[] sliceData = new Double[targetSize];
            for (int i = 0; i < targetSize; i++) {
                indices[dim] = i;
                sliceData[i] = source.item(indices);
            }

            Tensor<Double> sliceTensor = new Tensor<>(sliceData, targetSize);
            Tensor<Double> softmaxSlice = Softmax1d(sliceTensor);

            for (int i = 0; i < targetSize; i++) {
                indices[dim] = i;
                result.set(softmaxSlice.item(i), indices);
            }
            return;
        }

        for (int i = 0; i < shape[currentDim]; i++) {
            indices[currentDim] = i;
            applySoftmaxRecursive(source, result, dim, currentDim + 1, indices, shape);
        }
    }
}