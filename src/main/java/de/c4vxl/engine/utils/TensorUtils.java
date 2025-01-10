package de.c4vxl.engine.utils;

import de.c4vxl.engine.tensor.Tensor;
import de.c4vxl.engine.type.DType;
import de.c4vxl.engine.type.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

public class TensorUtils {
    /**
     * Fill the lower left triangle from a 2d tensor with a new value
     * @param tensor The tensor
     * @param newVal The new value
     */
    public static <T> Tensor<T> tril(Tensor<T> tensor, Object newVal) {
        if (tensor.shape.rank() != 2)
            throw new IllegalArgumentException("The tril operation only supports 2d tensors!");

        int rows = tensor.size(0);
        int cols = tensor.size(1);

        for (int i = 0; i < rows; i++)
            for (int j = i + 1; j < cols; j++)
                tensor.data[i * cols + j] = tensor.dtype.parse(newVal);

        return tensor;
    }

    /**
     * Applies a mask on a tensor
     * @param tensor The tensor to apply the mask onto
     * @param mask The mask to apply
     * @param checkFor The value in the mask to replace
     * @param value The value to replace masked items with
     */
    public static <T, R> Tensor<T> maskedFill(Tensor<T> tensor, Tensor<R> mask, Double checkFor, Double value) {
        return elementWise(tensor, mask.broadcastTo(tensor), (a, b) -> b.equals(checkFor) ? value : a);
    }

    /**
     * Returns a Tensor with all it's values filled with a specified value
     * @param tensor The tensor
     * @param obj The object to fill the tensors data with
     */
    public static <T> Tensor<T> filled(Tensor<T> tensor, Object obj) {
        Tensor<T> result = tensor.clone();
        Arrays.fill(result.data, result.dtype.parse(obj));
        return result;
    }

    /**
     * Perform an element wise operation between the values of two Tensors
     * @param a The first Tensor
     * @param b The second Tensor
     * @param operation The operation to perform. Format: (elementA, elementB) -> result
     */
    public static <T> Tensor<T> elementWise(Tensor<T> a, Tensor<?> b, BiFunction<Double, Double, Double> operation) {
        if (!a.shape.equals(b.shape)) {
            b = b.broadcastTo(a);
        }

//        if (!a.shape.equals(b.shape))
//            throw new IllegalArgumentException("Tensors a and b must be the same shape for element wise operations!");

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
     * Calculates all possible ways to index into a multidimensional shape
     * @param shape The shape
     */
    public static int[][] calculatePossibleIndices(int[] shape) {
        int[][] combinations = new int[TensorUtils.shapeToSize(shape)][shape.length];
        for (int i = 0; i < combinations.length; i++)
            combinations[i] = TensorUtils.unravelIndex(shape, i);
        return combinations;
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
     * Returns a narrowed version of the input tensor. The dimension dim is going to be the size (start...(start + length)).
     * @param tensor The input tensor
     * @param dim The dimension to narrow over
     * @param start The starting point
     * @param length The length of the narrowed window
     */
    public static <T> Tensor<T> narrow(Tensor<T> tensor, int dim, int start, int length) {
        dim = DataUtils.handleNegativeIndexing(tensor.shape.dimensions, dim);

        if (dim < 0 || dim >= tensor.shape.rank())
            throw new IndexOutOfBoundsException("Invalid dimension!");

        if (tensor.size(dim) < start + length || start < 0)
            throw new IndexOutOfBoundsException("Start index and length are out of bounds.");

        int[] newShape = tensor.shape.dimensions.clone();
        newShape[dim] = length;

        Tensor<T> result = new Tensor<>(tensor.dtype, newShape);
        for (int i = 0; i < result.size(); i++) {
            int[] idx = unravelIndex(result.shape.dimensions, i);
            idx[dim] += start;
            result.data[i] = tensor.data[flatIndex(tensor.shape.dimensions, idx)];
        }
        return result;
    }

    /**
     * Set a narrowed slice of the input tensor.
     * @param tensor The input tensor
     * @param dim The dimension to narrow over
     * @param start The starting point
     * @param slice The narrowed version to put in
     */
    public static <T> Tensor<T> narrow_set(Tensor<T> tensor, Tensor<T> slice, int dim, int start) {
        dim = DataUtils.handleNegativeIndexing(tensor.shape.dimensions, dim);

        if (dim < 0 || dim >= tensor.shape.rank())
            throw new IndexOutOfBoundsException("Invalid dimension!");
        if (start < 0 || tensor.size(dim) <= start)
            throw new IndexOutOfBoundsException("Start index is out of bounds.");

        if (slice.size(dim) != tensor.size(dim) - start)
            throw new IllegalArgumentException("Invalid slice length!");

        Tensor<T> result = tensor.clone();

        for (int i = 0; i < slice.data.length; i++) {
            int[] targetIdx = unravelIndex(slice.shape.dimensions, i);
            targetIdx[dim] += start;
            result = result.set(slice.data[i], targetIdx);
        }

        return result;
    }

    /**
     * Splits the tensor into chunks with individual sizes
     * @param tensor The tensor to split
     * @param dim The dimension to split over
     * @param chunkSizes The size for each chunk
     */
    @SuppressWarnings("unchecked")
    public static <T> Tensor<T>[] split(Tensor<T> tensor, int dim, int... chunkSizes) {
        if (Arrays.stream(chunkSizes).reduce(0, Integer::sum) != tensor.size(dim))
            throw new IllegalArgumentException("Chunk sizes don't add up to the size of the dimension!");
        
        Tensor<T>[] output = (Tensor<T>[]) new Tensor[chunkSizes.length];

        int start = 0;
        for (int i = 0; i < chunkSizes.length; i++) {
            output[i] = tensor.narrow(dim, start, chunkSizes[i]);

            start += chunkSizes[i];
        }
        
        return output;
    }

    /**
     * Split tensor into equally sized chunks of fixed size.
     * If the tensor size along the given dimension `dim` is not divisible by chunks, the last chunk will be the size of (dimSize % chunkSize).
     * @param tensor The tensor to chunk
     * @param dim The dimension to chunk over
     * @param chunkSize The size of each chunk
     */
    public static <T> Tensor<T>[] chunk(Tensor<T> tensor, int dim, int chunkSize) {
        int dimSize = tensor.size(dim);

        int[] chunkSizes = new int[(dimSize + chunkSize - 1) / chunkSize];
        chunkSizes[chunkSizes.length - 1] = (dimSize % chunkSize == 0) ? chunkSize : dimSize % chunkSize; // manually calculate last chunk
        for (int i = 0; i < chunkSizes.length - 1; i++)
            chunkSizes[i] = chunkSize;

        return TensorUtils.split(tensor, dim, chunkSizes);
    }

    /**
     * Concatenate a sequence of tensors along a new dimension
     * @param dim The index of where to add a dimension
     * @param tensors The tensors to stack
     */
    @SafeVarargs
    public static <T> Tensor<T> stack(int dim, Tensor<T>... tensors) {
        if (tensors.length == 0)
            throw new IllegalArgumentException("Must have at least one tensor.");

        Shape shape = tensors[0].shape;
        dim = DataUtils.handleNegativeIndexing(Arrays.copyOfRange(shape.dimensions, 0, shape.rank() + 1), dim);

        if (shape.rank() < dim)
            throw new IllegalArgumentException("Invalid dimension!");

        if (!Arrays.stream(tensors).allMatch(a -> a.shape.equals(shape)))
            throw new IllegalArgumentException("All tensors must be of the same shape!");

        List<Integer> newShape = new ArrayList<>(Arrays.stream(shape.dimensions).boxed().toList());
        newShape.add(dim, tensors.length);

        Tensor<T> result = new Tensor<>(tensors[0].dtype, newShape.stream().mapToInt(Integer::intValue).toArray());
        Integer[] index = new Integer[result.dim()];
        for (int i = 0; i < tensors.length; i++) {
            index[dim] = i;
            TensorUtils.setSlice(result, tensors[i].unsqueeze(0), index);
        }

        return result;
    }

    /**
     * Extract a slice out of a tensor based on given indices
     * @param tensor The tensor to extract from
     * @param dimensions The indices (int for specific dimension, "null" for selecting all elements across the dimension)
     *                   Missing dimensions will be padded with "null"-values
     * @param keepDims If a slice has empty dimensions and this value is set to "false", they will be removed
     */
    public static <T> Tensor<T> getSlice(Tensor<T> tensor, boolean keepDims, Integer... dimensions) {
        if (dimensions.length > tensor.shape.rank())
            throw new IndexOutOfBoundsException("Number of dimensions exceeds the tensor rank. (" + dimensions.length + " > " + tensor.shape.rank() + ")");

        dimensions = DataUtils.handleNegativeIndexing(tensor.shape.dimensions,
                DataUtils.padRight(dimensions, null, tensor.shape.dimensions.length, false));

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

        if (!keepDims)
            result = result.squeeze();

        return result;
    }

    /**
     * Set a slice/object of a Tensor based on given indices
     * @param tensor The tensor
     * @param dimensions The indices (int for specific dimension, "null" for selecting all elements across the dimension)
     *                   Missing dimensions will be padded with "null"-values
     * @param slice The slice to put in
     */
    public static <T> Tensor<T> setSlice(Tensor<T> tensor, Tensor<T> slice, Integer... dimensions) {
        if (dimensions.length > tensor.shape.rank())
            throw new IndexOutOfBoundsException("Number of dimensions exceeds the tensor rank. (" + dimensions.length + " > " + tensor.shape.rank() + ")");

        dimensions = DataUtils.handleNegativeIndexing(tensor.shape.dimensions,
                DataUtils.padRight(dimensions, null, tensor.shape.rank(), false));

        // add batch dimensions back to slice
        int[] sliceShape = TensorUtils.calculateSliceShape(tensor, dimensions);
        if (TensorUtils.shapeToSize(sliceShape) != slice.size())
            throw new IllegalArgumentException("Invalid slice shape! Expected " + Arrays.toString(sliceShape) + "!");
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

    /**
     * Reduce all elements across one dimension by applying a specific operation on them.
     * @param input The input tensor
     * @param dim The dimension to reduce over
     * @param operation The operation to use (for example: Tensor::add, Tensor::sub, ...)
     * @param keepDim Reducing over a dimension will result in the dimension becoming the size "1".
     *                If `keepDim` is false, this dimension will automatically be removed (squeezed).
     *                If `keepDim` is true, this dimension will be kept.
     */
    public static <T> Tensor<T> reduceAlongDimension(Tensor<T> input, int dim, BiFunction<Tensor<T>, Tensor<T>, Tensor<T>> operation, boolean keepDim) {
        dim = DataUtils.handleNegativeIndexing(input.shape.dimensions, dim);

        int[] outputShape = input.shape.dimensions.clone();
        outputShape[dim] = 1;
        Tensor<T> result = TensorUtils.filled(input.reshapeUnsafe(outputShape), 0).squeeze(dim);

        // perform operation
        // for i in 0...dimSize: operation( result, input[:, :, ..., i, :, :, ...] )
        Integer[] sliceIndex = new Integer[input.shape.rank()];
        for (int i = 0; i < input.size(dim); i++) {
            sliceIndex[dim] = i;
            result = operation.apply(result, input.get(sliceIndex));
        }

        if (!keepDim)
            result = result.unsqueeze(dim);

        return result;
    }

    /**
     * Perform matrix multiplication by splitting booth matrices into smaller blocks
     * @param a The first matrix
     * @param b The second matrix
     * @param result The container for the result
     * @param aRows Amount of rows in Tensor a (set to a.size(-2))
     * @param aCols Amount of columns in Tensor a (set to a.size(-1))
     * @param bCols Amount of columns in Tensor b (set to b.size(-1))
     * @param startRowA The starting point of the current batch (set to 0)
     * @param startColA The starting point of the current batch (set to 0)
     * @param startRowB The starting point of the current batch (set to 0)
     * @param startColB The starting point of the current batch (set to 0)
     * @param blockSize The size of the blocks
     */
    public static <T> void performBlockMultiplication(Tensor<T> a, Tensor<T> b, Tensor<T> result,
                                                      int aRows, int aCols, int bCols,
                                                      int startRowA, int startColA, int startRowB, int startColB,
                                                      int blockSize) {
        // base case: a and b are at block size -> multiply them
        if (aRows <= blockSize || aCols <= blockSize || bCols <= blockSize) {
            int[] batchShape = Arrays.copyOfRange(result.shape.dimensions, 0, result.shape.rank() - 2);

            for (int[] batchIndices : TensorUtils.calculatePossibleIndices(batchShape)) {
                Tensor<T> aSlice = TensorUtils.getSlice(a, false, DataUtils.IntegerIndex(batchIndices));
                Tensor<T> bSlice = TensorUtils.getSlice(b, false, DataUtils.IntegerIndex(batchIndices));
                Tensor<T> sliceResult = new Tensor<>(a.dtype, aRows, bCols);
                if (aSlice.shape.rank() == 0)
                    aSlice = aSlice.unsqueeze(0).unsqueeze(0);
                if (aSlice.shape.rank() == 1)
                    aSlice = aSlice.unsqueeze(0);
                if (bSlice.shape.rank() == 1)
                    bSlice = bSlice.unsqueeze(1);

                // perform matmul for this slice
                for (int i = 0; i < aRows; i++) {
                    for (int j = 0; j < bCols; j++) {
                        // calculate dot product for result[..., i, j]
                        double sum = 0;
                        for (int k = 0; k < aCols; k++)
                            sum = sum + aSlice.item(DType.DOUBLE, i, k) * bSlice.item(DType.DOUBLE, k, j);

                        sliceResult.set(a.dtype.parse(sum), i, j);
                    }
                }

                result.set(sliceResult, DataUtils.IntegerIndex(batchIndices));
            }

            return;
        }

        // or continue splitting them into smaller blocks

        int halfARows = aRows / 2;
        int halfACols = aCols / 2;
        int halfBCols = aCols / 2;

        // perform recursive block multiplication
        performBlockMultiplication(a, b, result, halfARows, halfACols, halfBCols, startRowA, startColA, startRowB, startColB, blockSize);
        performBlockMultiplication(a, b, result, halfARows, halfACols, halfBCols, startRowA, startColA + halfACols, startRowB + halfACols, startColB, blockSize);
        performBlockMultiplication(a, b, result, halfARows, halfACols, halfBCols, startRowA + halfARows, startColA, startRowB, startColB + halfBCols, blockSize);
        performBlockMultiplication(a, b, result, halfARows, halfACols, halfBCols, startRowA + halfARows, startColA + halfACols, startRowB + halfACols, startColB + halfBCols, blockSize);
    }
}