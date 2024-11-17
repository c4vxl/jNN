package de.c4vxl.engine.data;

import java.util.Arrays;

/**
 * Collection of utilities used by many components
 *
 * @author c4vxl
 */
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
     * @param checkFor The value to check in the mask
     * @param value The value to fill the masked items with
     */
    public static <T, R> Tensor<T> maskedFill(Tensor<T> tensor, Tensor<R> mask, R checkFor, T value) {
        // broadcast mask to shape of the tensor
        Tensor<R> broadcastedMask = Broadcasting.broadcastTo(mask, tensor.shape);

        // Apply mask
        for (int i = 0; i < tensor.data.length; i++) {
            int[] broadcastedIndices = Broadcasting.getBroadcastedIndices(i, tensor.shape, broadcastedMask.shape);
            int maskIndex = TensorUtils.computeBatchOffset(broadcastedIndices, broadcastedMask.shape);

            if (broadcastedMask.data[maskIndex].equals(checkFor)) {
                tensor.data[i] = value;
            }
        }

        return tensor;
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
     * Slice of the Tensor
     * @param tensor The Tensor
     * @param indices The indecision to slice along
     */
    public static <T> Tensor<T> slice(Tensor<T> tensor, int[] indices) {

        // create output Tensor
        Tensor<T> result = new Tensor<>(tensor.dtype,
                Arrays.copyOfRange(tensor.shape, indices.length, tensor.shape.length)); // calculate new shape

        // calculate offset
        int offset = 0;
        int stride = TensorUtils.shapeToSize(Arrays.copyOfRange(tensor.shape, indices.length, tensor.shape.length));
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * stride;
            stride /= tensor.shape[i];
        }

        // copy sliced data
        System.arraycopy(tensor.data, offset, result.data, 0, result.size);

        return result;
    }

    /**
     * Put a slice in the Tensor
     * @param tensor The Tensor
     * @param batchIdx The dimension to put it in
     * @param data The Slice
     */
    public static <T> void setSlice(Tensor<T> tensor, int batchIdx, T[] data) {
        int matrixSize = tensor.shape[tensor.shape.length - 2] * tensor.shape[tensor.shape.length - 1];
        if (data.length != matrixSize) {
            throw new IllegalArgumentException("Data size does not match matrix dimensions.");
        }

        int batchOffset = batchIdx * matrixSize;
        System.arraycopy(data, 0, tensor.data, batchOffset, matrixSize);
    }

    /**
     * Compute the offset for batching
     */
    public static int computeBatchOffset(int[] indices, int[] shape) {
        int offset = 0;
        int stride = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            offset += indices[i] * stride;
            stride *= shape[i];
        }

        return offset;
    }

    /**
     * Compute the stride
     */
    public static int computeStride(int[] shape, int dim) {
        int stride = 1;
        for (int i = dim + 1; i < shape.length; i++) {
            stride *= shape[i];
        }
        return stride;
    }
}