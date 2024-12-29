package de.c4vxl.engine.utils;

import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.type.DType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

public class TensorUtils {
    /**
     * Perform an element wise operation between the values of two Tensors
     * @param a The first Tensor
     * @param b The second Tensor
     * @param operation The operation to perform. Format: (elementA, elementB) -> result
     */
    public static <T> Tensor<T> elementWise(Tensor<T> a, Tensor<?> b, BiFunction<Double, Double, Double> operation) {
        if (!a.shape.equals(b.shape))
            b = b.broadcastTo(a);

        if (!a.shape.equals(b.shape))
            throw new IllegalArgumentException("Tensors a and b must be the same shape for element wise operations!");

        Tensor<?> finalB = b;
        return TensorUtils.elementWise(a.clone(), (elementA, index) ->
                operation.apply(
                        DType.DOUBLE.parse(elementA),
                        DType.DOUBLE.parse(finalB.data[index])
                )
        );
    }

    /**
     * Perform an operation on each element of a Tensor
     * @param tensor The Tensor
     * @param operation The operation to perform. Format: (element, flatIndex) -> result
     */
    public static <T> Tensor<T> elementWise(Tensor<T> tensor, BiFunction<T, Integer, Object> operation) {
        Tensor<T> result = tensor.clone();

        for (int i = 0; i < result.data.length; i++)
            result.data[i] = result.dtype.parse(operation.apply(result.data[i], i));

        return result;
    }

    /**
     * Returns the size of the shape if it was 1d
     * @param shape The shape
     */
    public static int shapeToSize(int... shape) { return Arrays.stream(shape).reduce(1, (a, b) -> a * b); }

    /**
     * Pad a shape by prepending ones to the left side
     * @param targetLength The targeted length
     * @param cut Define if the shape should be cut if it is already larger than `targetLength`
     * @param shape The shape to pad
     */
    public static int[] padShapeLeft(int targetLength, boolean cut, int... shape) {
        return DataUtils.intIndex(DataUtils.padLeft(DataUtils.IntegerIndex(shape), 1, targetLength, cut));
    }

    /**
     * Computes the strides of a shape
     * @param shape The shape
     */
    public static int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    /**
     * Convert a flat index into a multidimensional index for a given shape
     * @param shape The shape to index into
     * @param idx The flat index
     */
    public static int[] unravelIndex(int[] shape, int idx) {
        int[] result = new int[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            result[i] = idx % shape[i];
            idx /= shape[i];
        }
        return result;
    }

    /**
     * Flatten a multidimensional index
     * @param shape The shape of to index into
     * @param idx The multidimensional index
     */
    public static int flatIndex(int[] shape, int... idx) {
        idx = DataUtils.intIndex(DataUtils.handleNegativeIndexing(shape, DataUtils.IntegerIndex(idx)));

        int flatIdx = 0;
        for (int i = 0; i < idx.length; i++)
            flatIdx = flatIdx * shape[i] + idx[i];
        return flatIdx;
    }

    /**
     * Calculate the output shape of a slice
     * @param tensor The tensor to cut the slice out of
     * @param dimensions The position of the slice
     */
    public static int[] calculateSliceShape(Tensor<?> tensor, Integer[] dimensions) {
        // calculate new shape
        List<Integer> newShape = new ArrayList<>();
        for (int i = 0; i < dimensions.length; i++) {
            // if null: select all elements across the dimension
            if (dimensions[i] == null) newShape.add(tensor.size(i));

            // if out of bounds: throw error
            else if (dimensions[i] > tensor.size(i) || dimensions[i] < 0)
                throw new IndexOutOfBoundsException("Index " + dimensions[i] + " is out of bounds for dimension " + i + ".");

            // if not: add 1 (index points directly to an element)
            else newShape.add(1);
        }
        return newShape.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Extract a slice out of a tensor based on given indices
     * @param tensor The tensor to extract from
     * @param dimensions The indices (int for specific dimension, "null" for selecting all elements across the dimension)
     *                   Missing dimensions will be padded with "null"-values
     */
    public static <T> Tensor<T> index(Tensor<T> tensor, Integer... dimensions) {
        if (dimensions.length > tensor.shape.rank())
            throw new IndexOutOfBoundsException("Number of dimensions exceeds the tensor rank. (" + dimensions.length + " > " + tensor.shape.rank() + ")");


        dimensions = DataUtils.handleNegativeIndexing(tensor.shape.dimensions,
                DataUtils.padRight(dimensions, null, tensor.shape.rank(), false));

        Tensor<T> result = Tensor.empty(TensorUtils.calculateSliceShape(tensor, dimensions))
                .asDType(tensor.dtype);

        // populate result tensor
        for (int resultFlatIndex = 0; resultFlatIndex < result.data.length; resultFlatIndex++) {
            int[] resultIdx = unravelIndex(result.shape.dimensions, resultFlatIndex);

            int[] inputIdx = new int[tensor.shape.rank()];
            for (int i = 0; i < inputIdx.length; i++)
                inputIdx[i] = dimensions[i] == null ? resultIdx[i] : dimensions[i];

            result.data[resultFlatIndex] = tensor.data[flatIndex(tensor.shape.dimensions, inputIdx)];
        }

        return result.squeeze();
    }

    /**
     * Set a slice/object of a Tensor based on given indices
     * @param tensor The tensor
     * @param dimensions The indices (int for specific dimension, "null" for selecting all elements across the dimension)
     *                   Missing dimensions will be padded with "null"-values
     * @param slice The slice to put in
     */
    public static <T> Tensor<T> index_put(Tensor<T> tensor, Tensor<T> slice, Integer... dimensions) {
        if (dimensions.length > tensor.shape.rank())
            throw new IndexOutOfBoundsException("Number of dimensions exceeds the tensor rank. (" + dimensions.length + " > " + tensor.shape.rank() + ")");

        dimensions = DataUtils.handleNegativeIndexing(tensor.shape.dimensions,
                DataUtils.padRight(dimensions, null, tensor.shape.rank(), false));

        // add batch dimensions back to slice
        int[] sliceShape = TensorUtils.calculateSliceShape(tensor, dimensions);
        if (TensorUtils.shapeToSize(sliceShape) != slice.size())
            throw new IllegalArgumentException("Invalid slice shape!");
        slice = slice.reshape(sliceShape);

        // populate result tensor
        for (int sliceFlatIndex = 0; sliceFlatIndex < slice.data.length; sliceFlatIndex++) {
            int[] sliceIdx = unravelIndex(slice.shape.dimensions, sliceFlatIndex);

            int[] inputIdx = new int[tensor.shape.rank()];
            for (int i = 0; i < inputIdx.length; i++)
                inputIdx[i] = dimensions[i] == null ? sliceIdx[i] : dimensions[i];

            tensor.data[flatIndex(tensor.shape.dimensions, inputIdx)] = slice.data[sliceFlatIndex];
        }

        return tensor;
    }
}
