package de.c4vxl.engine.data;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Collection of utilities used by many components
 *
 * @author c4vxl
 */
@SuppressWarnings({"unchecked"})
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
     * @param dims The dimensions to index into
     * @param shape The dimensions
     */
    public static int[] handleNegativeDims(int[] dims, int... shape) {
        for (int i = 0; i < shape.length; i++) {
            shape[i] = shape[i] >= 0 ? shape[i] : dims.length + shape[i];
        }

        return shape;
    }

    /**
     * Allows for negative indexing into a shape
     * @param dims The dimensions to index into
     * @param pos The Position
     */
    public static int handleNegativeDim(int[] dims, int pos) {
        if (pos < 0) pos = dims.length + pos;

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
     * @param batchIdx The index to insert at
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
     * Compute a single stride across one dimension
     * @param shape The shape
     * @param dim The Dimension to take the stride from (can be negative)
     */
    public static int computeStride(int[] shape, int dim) {
        int[] strides = computeStrides(shape);
        return strides[handleNegativeDim(strides, dim)];
    }

    /**
     * Compute all strides across all dimensions
     * @param shape The shape
     */
    public static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    /**
     * Convert a flat index into a multi-dimensional index
     * @param shape The shape to index in
     * @param flatIndex The flat index
     */
    public static int[] computeIndices(int flatIndex, int[] shape) {
        int[] indices = new int[shape.length];
        int[] strides = computeStrides(shape);

        for (int i = 0; i < shape.length; i++) {
            indices[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }

        return indices;
    }

    /**
     * Flatten a multi-dimensional index
     * @param idx The multi-dimensional index (can be negative)
     * @param shape The shape to index into
     */
    public static int flatIndex(int[] shape, int... idx) {
        idx = TensorUtils.handleNegativeDims(shape, idx);

        int flatIndex = 0;
        for (int i = 0; i < idx.length; i++)
            flatIndex = flatIndex * shape[i] + idx[i];

        return flatIndex;
    }

    /**
     * Split a Tensor into multiple chunks over the last dimension
     * @param input The input tensor
     * @param chunkSize The size of the Tensor
     */
    public static <T> Tensor<T>[] chunk(Tensor<T> input, int chunkSize) { return chunk(input, chunkSize, -1); }

    /**
     * Split a Tensor into multiple chunks
     * @param input The input tensor
     * @param chunkSize The size of the Tensor
     * @param dim The dimension to chunk over
     */
    public static <T> Tensor<T>[] chunk(Tensor<T> input, int chunkSize, int dim) {
        // handle negative dims
        dim = handleNegativeDim(input.shape, dim);
        int dim_size = input.size(dim);

        // handle exceptions
        if (dim_size % chunkSize != 0)
            throw new IllegalArgumentException("Dimension size of target-dimension must be divisible by `chunkSize` for chunking!");

        // create output array
        int n_chunks = input.size(dim) / chunkSize;
        Tensor<T>[] outputs = new Tensor[n_chunks];

        // calculate shape of each chunk
        int[] output_shape = input.shape.clone();
        output_shape[dim] = chunkSize;

        // calculate scope details
        // a scope is one chunk of the input's data the size of chunkSize
        int[] output_shape_ = output_shape.clone();
        output_shape_[dim] = 1;
        int n_scopes = shapeToSize(output_shape_);
        int scope_distance = n_chunks * (chunkSize - 1);

        for (int chunk = 0; chunk < n_chunks; chunk++) {
            outputs[chunk] = new Tensor<>(input.dtype, output_shape);

            // calculate start and end of chunk scope
            int start = chunk * chunkSize;
            int end = (start + chunkSize * n_scopes + scope_distance * (n_scopes - 1)) - 2;

            // copy over data
            int chunkIndex = 0;
            for (int scopeStart = start; scopeStart < end; scopeStart+=(scope_distance + chunkSize)-1) {
                System.arraycopy(input.data, scopeStart, outputs[chunk].data, chunkIndex, chunkSize);
                chunkIndex += chunkSize;
            }
        }

        return outputs;
    }

    /**
     * Stack tensors on top of each other on a given dimension (the dimension will be created)
     * @param dim The dimension to stack on
     * @param tensors The list of the Tensors
     */
    public static <T> Tensor<T> stack(int dim, Tensor<T>... tensors) {
        // handle exceptions
        if (tensors.length == 0)
            throw new IllegalArgumentException("Must have at least one tensor!");

        int[] shape = tensors[0].shape;
        dim = handleNegativeDim(shape, dim);

        if (shape.length < dim)
            throw new IllegalArgumentException("Invalid dimension!");

        if (!Arrays.stream(tensors).allMatch((a) -> Arrays.equals(a.shape, shape)))
            throw new IllegalArgumentException("All tensors need to be the same shape to be stackable!");

        // at the index "dim" add a "1"
        Tensor<T> output = new Tensor<>(tensors[0].dtype,
                // calculate new shape with the extra dimension
                IntStream.concat(
                        IntStream.concat(Arrays.stream(Arrays.copyOfRange(shape, 0, dim)), IntStream.of(tensors.length)),
                        Arrays.stream(Arrays.copyOfRange(shape, dim, shape.length))
                ).toArray()
        );

        for (int i = 0; i < tensors.length; i++)
            System.arraycopy(tensors[i].data, 0, output.data, tensors[i].size * i, tensors[i].size);

        return output;
    }

    /**
     * Cut an input tensor to forget older tokens
     */
    public static <T> Tensor<T> cut_block_size(Tensor<T> input_ids, int block_size) {
        Tensor<T> input_ids_cut = input_ids.clone();
        if (block_size < input_ids_cut.size) {
            System.arraycopy(input_ids.data, input_ids_cut.size - block_size, input_ids_cut.data, 0, block_size);
            int[] shape = input_ids_cut.shape;
            shape[shape.length - 1] = block_size;
            input_ids_cut = input_ids_cut.reshapeUnsafe(shape);
        }
        return input_ids_cut;
    }

    /**
     * Generates samples from a multinomial distribution.
     * @param probs The probabilities tensor (1D tensor with probabilities summing to 1)
     * @param numSamples The number of samples to generate
     * @return A Tensor containing the generated samples
     */
    public static <T extends Number> Tensor<Integer> multinomial(Tensor<T> probs, int numSamples) {
        Tensor<Integer> result = new Tensor<>(DType.INTEGER, numSamples);
        Random rand = new Random();

        for (int i = 0; i < numSamples; i++) {
            double randomValue = rand.nextDouble();
            double cumulativeSum = 0.0f;

            for (int j = 0; j < probs.size(0); j++) {
                cumulativeSum += probs.data[j].doubleValue();
                if (randomValue < cumulativeSum) {
                    result.data[i] = result.valueOf(j);
                    break;
                }
            }
        }

        return result;
    }
}