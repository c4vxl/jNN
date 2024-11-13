package de.c4vxl.engine.data;

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
        System.out.println(booleanMask);
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
}
