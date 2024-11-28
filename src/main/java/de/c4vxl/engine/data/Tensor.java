package de.c4vxl.engine.data;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.*;
import java.util.function.BiFunction;

/**
 * A Tensor can be understood as a multidimensional matrix of different datatypes capable of performing various mathematical operations.
 * These operations include element-wise computation, matrix multiplication, reshaping, transposing, and more.
 *
 * @author c4vxl
 */
@SuppressWarnings("unchecked")
public class Tensor<T> {
    public int rank() { return this.shape.length; }
    public boolean is1d() { return rank() == 1; }
    public boolean is2d() { return rank() == 2; }
    public boolean is3d() { return rank() == 3; }

    public Class<T> dtype;
    public T[] data;
    public int[] shape;
    public int size;

    /**
     * Construct a Tensor with the default datatype
     * @param shape The shape of the Tensor
     */
    public Tensor(int... shape) {
        this((Class<T>) DType.DEFAULT, shape);
        this.randomized();
    }

    /**
     * Construct a Tensor with a specified datatype
     * @param dtype Data Type of the Tensor
     * @param shape The shape of the Tensor
     */
    public Tensor(Class<T> dtype, int... shape) {
        this((T[]) Array.newInstance(dtype, 1), shape);
        this.randomized();
    }

    /**
     * Construct a Tensor with preconfigured data
     * @param data Data of the Tensor
     * @param shape Shape of the Tensor
     */
    public Tensor(T[] data, int... shape) {
        this.shape = shape;
        this.size = TensorUtils.shapeToSize(shape);
        this.dtype = (Class<T>) data.getClass().getComponentType();

        // cut/loop data if it is too small/large
        this.data = (T[]) Array.newInstance(this.dtype, this.size);
        for (int i = 0; i < this.size; i++) {
            this.data[i] = data[i % data.length];
        }
    }

    /**
     * Construct a Tensor filled with one value repeatedly
     * @param obj The object the Tensor should be filled with
     * @param shape The shape of the Tensor
     */
    public static <T> Tensor<T> filled(T obj, int... shape) {
        return new Tensor<>((Class<T>) obj.getClass(), shape).fill(obj);
    }

    /**
     * Randomize this Tensor's values
     */
    public Tensor<T> randomized() {
        Tensor<T> tensor = this;

        Random rand = new Random();
        for (int i = 0; i < tensor.data.length; i++) {
            if (dtype == Double.class) tensor.data[i] = (T) Double.valueOf(rand.nextDouble());
            else if (dtype == Integer.class) tensor.data[i] = (T) Integer.valueOf(rand.nextInt(99));
            else if (dtype == Long.class) tensor.data[i] = (T) Long.valueOf(rand.nextLong());
            else if (dtype == Float.class) tensor.data[i] = (T) Float.valueOf(rand.nextFloat());
            else if (dtype == Boolean.class) tensor.data[i] = (T) Boolean.valueOf(rand.nextBoolean());
            else throw new IllegalArgumentException("Unsupported dtype '" + dtype.getSimpleName() + "'");
        }

        return tensor;
    }

    /**
     * Create a Tensor filled with random numbers
     * @param dtype Data type of the Tensor
     * @param shape Shape of the Tensor
     */
    public static <T> Tensor<T> random(Class<T> dtype, int... shape) {
        return new Tensor<>(dtype, shape).randomized();
    }

    /**
     * Create an empty Tensor with no shape and no data
     */
    public static <T> Tensor<T> empty(int... shape) {
        Tensor<T> tensor = Tensor.of();
        return shape.length >= 1 ? tensor.reshapeUnsafe(shape) : tensor;
    }

    /**
     * Construct a Tensor of a list of values
     * @param val List of values
     */
    public static <T> Tensor<T> of(T... val) {
        return new Tensor<>(val, val.length);
    }

    /**
     * Construct a Tensor filled with zeros
     * @param shape Shape of the Tensor
     */
    public static Tensor<Integer> zeros(int... shape) { return Tensor.filled(0, shape); }

    /**
     * Construct a Tensor filled with zeros
     * @param dtype Datatype of the Tensor
     * @param shape Shape of the Tensor
     */
    public static <T> Tensor<T> zeros(Class<T> dtype, int... shape) { return Tensor.filled(Objects.requireNonNull(DType.valueOf(dtype, "0")), shape); }

    /**
     * Construct a Tensor filled with ones
     * @param shape Shape of the Tensor
     */
    public static Tensor<Integer> ones(int... shape) { return Tensor.filled(1, shape); }

    /**
     * Construct a Tensor filled with ones
     * @param dtype Datatype of the Tensor
     * @param shape Shape of the Tensor
     */
    public static <T> Tensor<T> ones(Class<T> dtype, int... shape) { return Tensor.filled(Objects.requireNonNull(DType.valueOf(dtype, "1")), shape); }

    /**
     * Construct a Tensor filled with numbers of a range starting at 0 with step size = 1
     * @param end The end of the range
     */
    public static <T> Tensor<T> range(Class<T> dtype, int end) { return range(dtype, 0, end, 1); }

    /**
     * Construct a Tensor filled with numbers of a range
     * @param start The start of the range
     * @param end The end of the range
     * @param stepSize The size of the steps between start and end
     */
    public static <T> Tensor<T> range(Class<T> dtype, int start, int end, int stepSize) {
        if (start > end)
            throw new IllegalArgumentException("Start point can not be larger than end point!");

        end = start == 0 ? end - 1 : end;

        int size = (end - start) / stepSize + 1;

        Object[] data = (Object[]) Array.newInstance(dtype, size);

        for (int i = 0; i < size; i++)
            data[i] = DType.valueOf(dtype, "" + (start + i * stepSize));

        return new Tensor<>((T[]) data, size);
    }

    /**
     * Set a value of the data in this Tensor
     * @param obj The value to set
     * @param position The position for the value
     */
    public Tensor<T> set(T obj, int... position) {
        int idx = this.flatIndex(position);
        if (idx >= this.data.length)
            throw new IllegalArgumentException("Invalid position!");

        this.data[idx] = obj;

        return this;
    }

    /**
     * Get the size of one Dimension
     * @param pos The position
     */
    public int size(int pos) {
        return shape[TensorUtils.handleNegativeDim(this.shape, pos)];
    }

    /**
     * Fill the current Tensor with one value
     * @param obj The value to fill the Tensor with
     */
    public Tensor<T> fill(T obj) {
        Arrays.fill(this.data, obj);
        return this;
    }

    /**
     * Get the representation of a value in current dtype
     * @param val Your value to parse to the current dtype
     */
    public T valueOf(Object val) { return DType.valueOf(dtype, val); }

    /**
     * Flatten the index of the location in the Tensor
     * @param idx Location in the Tensor
     */
    public int flatIndex(int... idx) { return TensorUtils.flatIndex(this.shape, idx); }

    /**
     * Get the item in this Tensor
     * @param loc Location of the item, if not passed, the first item will be used
     */
    public T item(int... loc) {
        return data[loc.length > 0 ? flatIndex(loc) : 0];
    }

    /**
     * Get the item in this Tensor as a specific data type
     * @param loc Location of the item, if not passed, the first item will be used
     * @param dtype The new data type
     */
    public <R> R item(Class<R> dtype, int... loc) {
        return DType.valueOf(dtype, item(loc));
    }

    /**
     * Create a copy of this Tensor with another data type
     * @param dtype The new data type
     */
    public <R> Tensor<R> asDType(Class<R> dtype) {
        if (dtype == this.dtype) return (Tensor<R>) this;

        R[] data = (R[]) Array.newInstance(dtype, this.size);
        for (int i = 0; i < data.length; i++) {
            data[i] = DType.valueOf(dtype, this.data[i]);
        }

        return new Tensor<>(data, this.shape);
    }

    /**
     * Sum across a given dimension.
     */
    public Tensor<T> sum(int axis) {
        if (axis < 0 || axis >= this.shape.length)
            throw new IllegalArgumentException("Invalid axis specified.");

        if (dtype == Boolean.class)
            throw new RuntimeException("Operation cannot be performed on dtype 'Boolean'");

        int[] newShape = shape.length == 1 ? new int[]{shape[0]} : new int[shape.length - 1];

        int newSize = 1;
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                newShape[j++] = shape[i];
                newSize *= shape[i];
            }
        }

        // Summing across the specified axis
        T[] resultData = (T[]) Array.newInstance(dtype, newSize);
        Arrays.fill(resultData, valueOf(0)); // Start with zeros for sum accumulation
        int[] index = new int[shape.length];
        for (int i = 0; i < size; i++) {
            int currentIndex = i;

            // Compute the index array for the current element
            for (int j = shape.length - 1; j >= 0; j--) {
                index[j] = currentIndex % shape[j];
                currentIndex /= shape[j];
            }

            // Determine output index based on the axis we're summing over
            int outputIndex = 0;
            for (int j = 0, k = 0; j < shape.length; j++) {
                if (j != axis) {
                    outputIndex = outputIndex * newShape[k] + index[j];
                    k++;
                }
            }

            // Perform the summation along the specified axis
            resultData[outputIndex] = numericalOperation(resultData[outputIndex], data[i], Double::sum);
        }

        return new Tensor<>(resultData, newShape);
    }

    /**
     * Perform an operation on each element in this Tensor
     */
    public Tensor<T> elementWise(BiFunction<T, Integer, Object> task) {
        Tensor<T> out = this.clone();
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = valueOf(task.apply(this.data[i], i));
        }

        return out;
    }

    /**
     * Perform an operation element wise with another Tensor
     * @param other The second Tensor
     * @param function The function to apply
     */
    public Tensor<T> elementWise(Tensor<T> other, BiFunction<Double, Double, Double> function) {
        // broadcast tensors
        other = Broadcasting.broadcastTo(other, this.shape);

        Tensor<T> result = this.clone();
        for (int i = 0; i < other.size; i++) {
            result.data[i] = valueOf( // convert result back to dtype of this Tensor
                    function.apply(
                            DType.valueOf(Double.class, result.data[i]), // convert a to double
                            DType.valueOf(Double.class, other.data[i])  // convert b to double
                    )
            );
        }
        return result;
    }

    /**
     * Perform a numerical operation between two values
     */
    public T numericalOperation(T a, T b, BiFunction<Double, Double, Double> operation) {
        if (dtype == Boolean.class)
            throw new RuntimeException("Operation can not be performed on dtype 'Boolean'");

        Double result = operation.apply(((Number) a).doubleValue(), ((Number) b).doubleValue());
        return valueOf(result.toString());
    }

    /**
     * Compute the exponential of each element
     */
    public Tensor<T> exp() { return this.clone().elementWise((a, i) -> Math.exp(((Number) a).doubleValue())); }

    /**
     * Perform element wise addition
     * @param other Pass the other object
     */
    public Tensor<T> add(T other) {
        return this.add(Tensor.filled(other, shape));
    }

    /**
     * Perform element wise addition
     * @param other Pass the other tensor
     */
    public Tensor<T> add(Tensor<T> other) {
        return this.clone().elementWise(other, Double::sum);
    }

    /**
     * Perform element wise subtraction
     * @param other Pass the other object
     */
    public Tensor<T> sub(T other) {
        return this.sub(Tensor.filled(other, shape));
    }

    /**
     * Perform element wise subtraction
     * @param other Pass the other tensor
     */
    public Tensor<T> sub(Tensor<T> other) {
        return this.clone().elementWise(other, (a, b) -> a - b);
    }

    /**
     * Perform element wise division
     * @param other Pass the other object
     */
    public Tensor<T> div(T other) {
        return this.div(Tensor.filled(other, shape));
    }

    /**
     * Perform element wise division
     * @param other Pass the other tensor
     */
    public Tensor<T> div(Tensor<T> other) {
        return this.clone().elementWise(other, (a, b) -> a / b);
    }

    /**
     * Perform element wise multiplication
     * @param other Pass the other object
     */
    public Tensor<T> mul(T other) {
        return this.mul(Tensor.filled(other, shape));
    }

    /**
     * Perform element wise multiplication
     * @param other Pass the other tensor
     */
    public Tensor<T> mul(Tensor<T> other) {
        return this.clone().elementWise(other, (a, b) -> a * b);
    }

    /**
     * Perform element wise square root
     */
    public Tensor<T> sqrt() {
        return this.clone().elementWise((a, i) ->
                Math.sqrt(((Number) a).doubleValue())
        );
    }

    /**
     * Perform element wise power
     */
    public Tensor<T> pow(Number power) {
        return this.clone().elementWise((a, i) ->
                Math.pow(((Number) a).doubleValue(), power.doubleValue())
        );
    }

    /**
     * Perform element logarithm
     */
    public Tensor<T> log() {
        return this.clone().elementWise((a, i) ->
                Math.log(((Number) a).doubleValue())
        );
    }

    /**
     * Get the smallest object in the Tensor
     */
    public T min() {
        return Arrays.stream(this.data).min((Comparator<? super T>) Comparator.naturalOrder()).orElseThrow();
    }

    /**
     * Get the largest object in the Tensor
     */
    public T max() {
        return Arrays.stream(this.data).max((Comparator<? super T>) Comparator.naturalOrder()).orElseThrow();
    }

    /**
     * Fill this Tensor with items
     * @param items Pass the items to put into the Tensor's data
     */
    public Tensor<T> fillWith(T... items) {
        if(items.length > this.data.length)
            throw new IllegalArgumentException("Items can not be more than the Tensor can hold!");

        System.arraycopy(items, 0, this.data, 0, items.length);

        return this;
    }

    /**
     * Clip each element at a min and max
     * @param max The highest value an element can reach
     * @param min The lowest value an element can reach
     */
    public Tensor<T> clip(T min, T max) {
        if (dtype == Boolean.class)
            throw new RuntimeException("Operation can not be performed on dtype 'Boolean'");

        return this.clone().elementWise((a, index) -> {
            a = numericalOperation(a, max, (s, m) -> s > m ? m : s); // clip maximum
            return numericalOperation(a, min, (s, m) -> s < m ? m : s); // clip minimum
        });
    }

    /**
     * Reshape the Tensor
     * @param shape New shape of the Tensor
     */
    public Tensor<T> reshape(int... shape) {
        shape = TensorUtils.handleNegativeDims(this.shape, shape);

        if (TensorUtils.shapeToSize(shape) != this.size)
            throw new IllegalArgumentException("New shape must still be the same size as the old one!");

        return reshapeUnsafe(shape);
    }

    /**
     * Reshape a Tensor without checking if the new shape is possible.
     * This allows for expanding / shrinking the size of a Tensor.
     * WARNING: Might cause null-Values in this.data
     * @param shape New shape of the Tensor
     */
    public Tensor<T> reshapeUnsafe(int... shape) {
        Tensor<T> res = this.clone();
        res.shape = TensorUtils.handleNegativeDims(this.shape, shape);
        res.data = (T[]) Array.newInstance(dtype, TensorUtils.shapeToSize(shape));
        for (int i = 0; i < res.data.length; i++) {
            res.data[i] = this.data.length > i ? this.data[i] : valueOf(0);
        }
        res.size = TensorUtils.shapeToSize(shape);
        return res;
    }

    /**
     * Insert a Dimension at a given position
     * @param pos The position to insert a dimension at
     */
    public Tensor<T> unsqueeze(int pos) {
        if (pos < 0) pos = this.shape.length + pos; // handle negative indexes

        if (this.shape.length <= pos)
            throw new IllegalArgumentException("Invalid position!");

        // add dim
        ArrayList<Integer> shape = new ArrayList<>(Arrays.stream(this.shape).boxed().toList());
        shape.add(pos, 1);

        return this.reshape(shape.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Remove a Dimension at a given position
     * @param pos The position to remove a dimension at
     */
    public Tensor<T> squeeze(int pos) {
        if (pos < 0) pos = this.shape.length + pos; // handle negative indexes

        if (this.shape.length <= pos)
            throw new IllegalArgumentException("Invalid position!");

        if (this.shape[pos] > 1)
            throw new IllegalArgumentException("Can only squeeze dimensions the size of 1!");

        // remove dim
        ArrayList<Integer> shape = new ArrayList<>(Arrays.stream(this.shape).boxed().toList());
        shape.remove(pos);

        return this.reshape(shape.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Perform matrix multiplication (or batched matrix multiplication if needed) with another Tensor
     * @param b The second Tensor
     */
    public Tensor<T> matmul(Tensor<T> b) {
        if (this.shape.length < 1 || b.shape.length < 1)
            throw new IllegalArgumentException("Matmul requires tensors with at least one dimension.");

        Tensor<T> a = this;

        // reshape 1D tensors to 2D for matmul
        boolean aWas1D = false, bWas1D = false;
        if (a.shape.length == 1) {
            a = a.unsqueeze(0); // Prepend 1 to dimension
            aWas1D = true;
        }
        if (b.shape.length == 1) {
            b = b.unsqueeze(-1); // Append 1 to dimension
            bWas1D = true;
        }

        // broadcast dims
        int[] batchShapeA = Arrays.copyOfRange(a.shape, 0, a.shape.length - 2);
        int[] batchShapeB = Arrays.copyOfRange(b.shape, 0, b.shape.length - 2);
        int[] broadcastedBatchShape = Broadcasting.broadcastShapes(batchShapeA, batchShapeB);


        int aMatrixRows = a.size(-2);
        int aMatrixCols = a.size(-1);
        int bMatrixCols = b.size(-1);

        // calculate result shape
        int[] resultShape = Arrays.copyOf(broadcastedBatchShape, broadcastedBatchShape.length + 2);
        resultShape[resultShape.length - 2] = aMatrixRows;
        resultShape[resultShape.length - 1] = bMatrixCols;

        // perform batched matmul
        Tensor<T> result = this.clone().reshapeUnsafe(resultShape);
        int batchSize = TensorUtils.shapeToSize(broadcastedBatchShape);

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            int[] aBatchIndices = Broadcasting.getBroadcastedIndices(batchIdx, batchShapeA, broadcastedBatchShape);
            int[] bBatchIndices = Broadcasting.getBroadcastedIndices(batchIdx, batchShapeB, broadcastedBatchShape);

            Tensor<T> aMatrix = TensorUtils.slice(a, aBatchIndices);
            Tensor<T> bMatrix = TensorUtils.slice(b, bBatchIndices);

            T[] resultMatrixData = (T[]) Array.newInstance(dtype, aMatrixRows * bMatrixCols);

            for (int i = 0; i < aMatrixRows; i++) {
                for (int j = 0; j < bMatrixCols; j++) {
                    T sum = valueOf("0");
                    for (int k = 0; k < aMatrixCols; k++) {
                        T aValue = aMatrix.item(i, k);
                        T bValue = bMatrix.item(k, j);
                        sum = numericalOperation(sum, numericalOperation(aValue, bValue, (x, y) -> x * y), Double::sum);
                    }
                    resultMatrixData[i * bMatrixCols + j] = sum;
                }
            }

            TensorUtils.setSlice(result, batchIdx, resultMatrixData);
        }

        // remove added dims
        if (aWas1D)
            result = result.squeeze(0);

        if (bWas1D)
            result = result.squeeze(-1);

        return result;
    }

    /**
     * concatenate two Tensors. First available axis will be used!
     * @param other The other Tensor
     */
    public Tensor<T> concatenate(Tensor<T> other) {
        if (this.size != other.size)
            throw new IllegalArgumentException("Tensors must have the same number of dimensions.");

        // find first matching axis
        int axis = -1;
        for (int i = 0; i < this.shape.length; i++) {
            if (this.shape[i] == other.shape[i]) {
                axis = i;
                break;
            }
        }

        if (axis == -1)
            throw new IllegalArgumentException("No matching axis found for concatenation.");

        Tensor<T> result = this.clone();

        // concatenate shapes
        int newSize = this.shape[axis] + other.shape[axis];
        int totalSize = result.data.length + other.data.length;

        // new data
        T[] concatenatedData = Arrays.copyOf(result.data, totalSize);
        System.arraycopy(other.data, 0, concatenatedData, result.data.length, other.data.length);
        result.data = concatenatedData;

        // update shape
        result.shape[axis] = newSize;
        result.reshapeUnsafe(shape);

        return result;
    }

    /**
     * Transpose the Tensor's data over different dimensions
     * @param dims The dimensions to transpose the Tensor over
     */
    public Tensor<T> transpose(int... dims) {
        // just flip if it's a 2d Tensor and no dims are specified
        if (dims.length == 0 && is2d()) dims = new int[]{1, 0};

        if (dims.length == 0 || dims.length != shape.length)
            throw new IllegalArgumentException("Invalid dimensions specified!");

        // handle negative dims
        dims = TensorUtils.handleNegativeDims(this.shape, dims);

        // calculate new shape and new data array
        int[] newShape = Arrays.stream(dims).map(d -> shape[d]).toArray();
        T[] resultData = (T[]) Array.newInstance(data.getClass().getComponentType(),
                TensorUtils.shapeToSize(shape));

        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];

        for (int i = 0; i < resultData.length; i++) {
            int[] newIndex = new int[newShape.length];
            int temp = i;
            for (int j = newShape.length - 1; j >= 0; j--) {
                newIndex[j] = temp % newShape[j];
                temp /= newShape[j];
            }

            int originalIndex = 0;
            for (int j = 0; j < dims.length; j++)
                originalIndex += newIndex[j] * strides[dims[j]];

            resultData[i] = data[originalIndex];
        }

        return new Tensor<>(resultData, newShape);
    }

    @Override
    public Tensor<T> clone() {
        return new Tensor<>(data.clone(), shape.clone());
    }

    @Override
    public String toString() {
        return "Tensor{" +
                "dtype=" + dtype.getSimpleName() +
                ", shape=" + Arrays.toString(shape.clone()) +
                ", size=" + size +
                ", data=" + Arrays.toString(data.clone()) +
                '}';
    }
}