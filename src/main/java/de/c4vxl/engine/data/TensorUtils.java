package de.c4vxl.engine.data;

import java.lang.reflect.Array;
import java.util.Arrays;

public class TensorUtils {
    /**
     * Create a lower triangular Tensor from the current Tensor. The rest will be set to 0
     * @param tensor The tensor to operate on
     */
    public static <T> Tensor<T> tril(Tensor<T> tensor) {
        return tril(tensor, 0);
    }

    /**
     * Create a lower triangular Tensor from the current Tensor.
     * @param tensor The tensor to operate on
     * @param newVal The value the upper triangle should be set to
     */
    public static <T> Tensor<T> tril(Tensor<T> tensor, Object newVal) {
        if (!tensor.is2d())
            throw new IllegalArgumentException("The 'tril' operation is only supported for 2D tensors.");

        int rows = tensor.size(0);
        int cols = tensor.size(1);

        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                tensor.data[i * cols + j] = tensor.valueOf(newVal);
            }
        }

        return tensor;
    }

    /**
     * Apply a mask to a tensor
     * @param tensor The Tensor to operate on
     * @param mask The mask
     * @param value The value to fill the masked items with
     */
    public static <T, R> Tensor<T> maskedFill(Tensor<T> tensor, Tensor<R> mask, R checkFor, T value) {
        if (!Arrays.equals(tensor.shape, mask.shape))
            throw new IllegalArgumentException("Mask and tensor must have the same shape!");

        if (!Number.class.isAssignableFrom(mask.dtype))
            throw new IllegalArgumentException("Mask must have dtype Boolean or a numeric type!");

        Tensor<Boolean> booleanMask = mask.asDType(Boolean.class);
        for (int i = 0; i < tensor.data.length; i++) {
            if (booleanMask.data[i] == Tensor.valueOf(Boolean.class, checkFor)) {
                tensor.data[i] = tensor.valueOf(value);
            }
        }

        return tensor;
    }

    /**
     * Remove the batch dimensions (size=1) from the Tensor
     * @param tensor The Tensor to operate on
     */
    public static <T> Tensor<T> withoutBatchDim(Tensor<T> tensor) {
        return tensor.reshape(Arrays.stream(tensor.shape)
                .filter(num -> num != 1)  // Filter out 1
                .toArray());
    }

    /**
     * Allows for negative indexing into the shape of a Tensor
     * @param tensor The Tensor to index into
     * @param shape The dimensions
     */
    public static int[] handleNegativeDims(Tensor<?> tensor, int... shape) {
        for (int i = 0; i < shape.length; i++) {
            shape[i] = shape[i] >= 0 ? shape[i] : tensor.shape.length + shape[i];
        }

        return shape;
    }

    /**
     * Allows for negative indexing into the shape of a Tensor
     * @param tensor The Tensor to index into
     * @param pos The Position
     */
    public static int handleNegativeDim(Tensor<?> tensor, int pos) {
        if (pos < 0) pos = tensor.shape.length + pos;

        return pos;
    }

    /**
     * Calculate the final size of a Tensor out of it's shape
     * @param shape The shape of the Tensor
     */
    public static int shapeToSize(int... shape) {
        return Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    }

    /**
     * Perform batched matrix multiplication with another Tensor.
     * @param a The Tensor
     * @param b The Tensor to multiply with
     */
    public static  <T> Tensor<T> matmul_batched(Tensor<T> a, Tensor<T> b) {
        if (a.size(0) != b.shape[0])
            throw new IllegalArgumentException("Incompatible batch sizes: " +
                    Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape));

        int batchSize = a.size(0); // b
        int m = a.size(-2);
        int n = a.size(-1);
        int p = b.size(-1);

        int[] resultShape = a.shape.clone();
        resultShape[resultShape.length-1] = p;
        resultShape[resultShape.length-2] = m;

        Tensor<T> result = a.clone();
        T[] resultData = (T[]) Array.newInstance(a.dtype, shapeToSize(resultShape));

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < a.shape[1]; j++) {
                for (int k = 0; k < m; k++) {
                    for (int l = 0; l < p; l++) {
                        T sum = a.valueOf("0");

                        for (int t = 0; t < n; t++) {
                            T s = a.data[i * a.shape[1] * m * n + j * m * n + k * n + t]; // element from a
                            T bValue = b.data[i * b.shape[1] * n * p + j * n * p + t * p + l];  // element from b

                            // sum products
                            sum = a.numericalOperation(sum, a.numericalOperation(s, bValue, (x, y) -> x * y), Double::sum);
                        }

                        // store result in result Tensor
                        resultData[i * a.shape[1] * m * p + j * m * p + k * p + l] = sum;
                    }
                }
            }
        }

        // overwrite data and resize
        result.data = resultData;
        return result.reshapeUnsafe(resultShape);
    }
}