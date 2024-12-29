package de.c4vxl.engine.tensor;

import de.c4vxl.engine.type.DType;
import de.c4vxl.engine.type.Shape;
import de.c4vxl.engine.utils.BroadcastingUtils;
import de.c4vxl.engine.utils.DataUtils;
import de.c4vxl.engine.utils.TensorUtils;

import java.lang.reflect.Array;
import java.util.*;

@SuppressWarnings("unchecked")
public class Tensor<T> {
    public T[] data;
    public Shape shape;
    public DType<T> dtype;

    /**
     * Get the size of this Tensors data
     */
    public int size() { return this.shape.size(); }

    /**
     * Get the size of a specific dimension of this Tensors shape
     */
    public int size(int dim) { return this.shape.dimensions[DataUtils.handleNegativeIndexing(this.shape.dimensions, dim)]; }

    /**
     * Construct a Tensor filled with random values between 0 and 1 with a given shape and DEFAULT dtype
     * @param shape The shape of the Tensor
     */
    public Tensor(int... shape) { this((DType<T>) DType.DEFAULT, shape); }

    /**
     * Construct a Tensor filled with random values between 0 and 1 with a given dtype and shape
     * @param dtype The data type of the Tensor
     * @param shape The shape of the Tensor
     */
    public Tensor(DType<T> dtype, int... shape) {
        this((T[]) Array.newInstance(dtype.clazz, 1), shape);

        // randomize data
        DataUtils.randomInitialization(this.data);
    }

    /**
     * Construct a Tensor with a given data array and a given shape
     * @param data The data for the Tensor.
     *             If it doesn't fit the size specified with the `shape` argument, the data might be cut/looped!
     * @param shape The shape of the Tensor.
     */
    public Tensor(T[] data, int... shape) {
        this.dtype = (DType<T>) new DType<>(data.getClass().getComponentType());
        this.shape = new Shape(shape);

        // loop/cut data
        this.data = (T[]) Array.newInstance(this.dtype.clazz, this.shape.size());
        for (int i = 0; i < this.data.length; i++)
            this.data[i] = data[i % data.length];
    }

    /**
     * Construct a Tensor filled with random values in between two bounds
     * @param dtype The data type of the Tensor
     * @param min The min value
     * @param max The max value
     * @param shape The shape of the Tensor
     */
    public static <T> Tensor<T> random(DType<T> dtype, double min, double max, int... shape) {
        Tensor<T> tensor = new Tensor<>(dtype, shape);

        for (int i = 0; i < tensor.data.length; i++)
            tensor.data[i] = dtype.randomValue(min, max);

        return tensor;
    }

    /**
     * Construct a Tensor filled with random integers in between two bounds
     * @param min The min value
     * @param max The max value
     * @param shape The shape of the Tensor
     */
    public static Tensor<Integer> randint(int min, int max, int... shape) { return Tensor.random(DType.INTEGER, min, max, shape); }

    /**
     * Construct a fully empty Tensor with null values
     * @param shape The shape of the Tensor
     */
    public static Tensor<?> empty(int... shape) { return new Tensor<>(new Object[]{ null }, shape); }

    /**
     * Construct a 1d Tensor with a given list of data
     * @param data The data for the Tensor
     */
    public static <T> Tensor<T> of(T... data) {
        return new Tensor<>(data, data.length);
    }

    /**
     * Construct a Tensor filled with one repeated value
     * @param obj The object to fill the Tensor with
     * @param shape The shape of the Tensor
     */
    public static <T> Tensor<T> filled(T obj, int... shape) {
        T[] data = (T[]) Array.newInstance(obj.getClass(), 1);
        data[0] = obj;
        return new Tensor<>(data, shape);
    }

    /**
     * Construct a Tensor filled with zeros
     * @param shape The shape of the Tensor
     */
    public static Tensor<Integer> zeros(int... shape) { return Tensor.filled(0, shape); }

    /**
     * Construct a Tensor filled with ones
     * @param shape The shape of the Tensor
     */
    public static Tensor<Integer> ones(int... shape) { return Tensor.filled(1, shape); }

    /**
     * Construct an Integer-tensor with values from `0`...`size(shape)`
     * @param shape The shape of the Tensor
     */
    public static Tensor<Integer> range(int... shape) {
        return Tensor.range(DType.INTEGER, 0, TensorUtils.shapeToSize(shape), 1).reshape(shape);
    }

    /**
     * Construct a 1-D tensor with values from the interval `start`...`end` with a given gap between two adjacent values
     * @param dtype The data type of the Tensor
     * @param start The starting point
     * @param end The ending point + 1
     * @param stepSize The gap in between two neighbouring values
     */
    public static <T> Tensor<T> range(DType<T> dtype, int start, int end, int stepSize) {
        if (start >= end)
            throw new IllegalArgumentException("The starting point of the range must be smaller than the ending point! (" + start + " >= " + end + ")");

        end -= 1;

        T[] data = (T[]) Array.newInstance(dtype.clazz, (end - start) / stepSize + 1);
        for (int i = 0; i < data.length; i++)
            data[i] = dtype.parse(start + i * stepSize);

        return new Tensor<>(data, data.length);
    }

    /**
     * Returns a copy of this Tensor as DType.DOUBLE
     */
    public Tensor<Double> asDouble() { return this.asDType(DType.DOUBLE); }

    /**
     * Returns a copy of this Tensor as DType.FLOAT
     */
    public Tensor<Float> asFloat() { return this.asDType(DType.FLOAT); }

    /**
     * Returns a copy of this Tensor as DType.BOOLEAN
     */
    public Tensor<Boolean> asBool() { return this.asDType(DType.BOOLEAN); }

    /**
     * Returns a copy of this Tensor as DType.INTEGER
     */
    public Tensor<Integer> asInt() { return this.asDType(DType.INTEGER); }

    /**
     * Returns a copy of this Tensor as DType.LONG
     */
    public Tensor<Long> asLong() { return this.asDType(DType.LONG); }

    /**
     * Returns a copy of this Tensor as another dtype
     * @param target The targeted dtype
     */
    public <R> Tensor<R> asDType(DType<R> target) {
        R[] data = (R[]) Array.newInstance(target.clazz, this.size());
        for (int i = 0; i < data.length; i++)
            data[i] = target.parse(this.data[i]);

        return new Tensor<>(data.clone(), this.shape.dimensions.clone());
    }

    /**
     * Reshape this Tensor to a new shape
     * @param newShape The new shape of the Tensor
     */
    public Tensor<T> reshape(int... newShape) {
        if (TensorUtils.shapeToSize(newShape) != this.shape.size())
            throw new IllegalArgumentException("The size of a Tensor can't change while reshaping!");

        return this.reshapeUnsafe(newShape);
    }

    /**
     * Reshape this Tensor to a new shape whiles ignoring the size boundaries
     * @param newShape The new shape of the Tensor
     */
    public Tensor<T> reshapeUnsafe(int... newShape) {
        Tensor<T> result = this.clone();
        result.shape = new Shape(newShape);
        result.data = (T[]) Array.newInstance(this.dtype.clazz, result.shape.size());
        for (int i = 0; i < result.data.length; i++)
            result.data[i] = this.data.length > i ? this.data[i] : null;
        return result;
    }

    /**
     * Insert an "empty" dimension of size "1" in a given position between (-rank-1...rank)
     * @param dim The index to insert dimension at
     */
    public Tensor<T> unsqueeze(int dim) {
        dim = DataUtils.handleNegativeIndexing(Arrays.copyOfRange(this.shape.dimensions, 0, this.shape.rank() + 1), dim);

        if (this.shape.rank() < dim || dim < 0)
            throw new IndexOutOfBoundsException("Invalid position! Only in range of (" + -(this.shape.rank() + 1) + "..." + this.shape.rank() + ").");

        List<Integer> shape = new ArrayList<>(Arrays.stream(this.shape.dimensions).boxed().toList());
        shape.add(dim, 1);
        return this.reshape(shape.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Remove all "empty" dimension of size "1"
     */
    public Tensor<T> squeeze() {
        List<Integer> shape = new ArrayList<>(Arrays.stream(this.shape.dimensions).boxed().toList());
        shape.removeIf((x) -> x == 1);
        return this.reshape(shape.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Remove an "empty" dimension of size "1" in a given position between (-rank+1...rank)
     * @param dim The index to insert dimension at
     */
    public Tensor<T> squeeze(int dim) {
        dim = DataUtils.handleNegativeIndexing(this.shape.dimensions, dim);

        if (this.shape.rank() < dim || dim < 0)
            throw new IndexOutOfBoundsException("Invalid position! Only in range of (" + -this.shape.rank() + "..." + this.shape.rank() + ").");

        if (this.shape.dimensions[dim] != 1)
            throw new IllegalArgumentException("Can only squeeze dimensions of size 1");

        List<Integer> shape = new ArrayList<>(Arrays.stream(this.shape.dimensions).boxed().toList());
        shape.remove(dim);
        return this.reshape(shape.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Broadcast this Tensor to another shape
     * More information at <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">Numpy broadcasting docs</a>
     *
     * @param shape The shape to broadcast the Tensor to
     */
    public Tensor<T> broadcastTo(int... shape) {
        int[] broadcastedShape = BroadcastingUtils.broadcastShapes(this.shape.dimensions, shape);
        Tensor<T> result = this.clone().reshapeUnsafe(broadcastedShape);
        result.data = BroadcastingUtils.broadcastData(this.data, this.shape.dimensions, broadcastedShape);
        return result;
    }

    /**
     * Broadcast this Tensor to the shape of another Tensor
     * More information at <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">Numpy broadcasting docs</a>
     *
     * @param other The other tensor
     */
    public Tensor<T> broadcastTo(Tensor<?> other) { return this.broadcastTo(other.shape.dimensions); }

    /**
     * Index into the Tensors data with a multidimensional index
     * more information about indexing at `TensorUtils.index`
     *
     * @param idx The index
     * @return If the index points to a direct element, the element will be returned wrapped in a 1d tensor,
     *         if it points to a slice, the slice will be returned
     */
    public Tensor<T> get(Integer... idx) {
        // points to a direct item
        if (idx.length == this.shape.rank() && !Arrays.stream(idx).toList().contains(null))
            return Tensor.of(this.item(DataUtils.intIndex(idx)));

        // points to a slice
        return TensorUtils.index(this, idx);
    }

    /**
     * Get a raw item at a position in the tensor
     * @param idx The index of the item. Can be flat or multidimensional.
     */
    public T item(int... idx) { return this.item(this.dtype, idx); }

    /**
     * Get a raw item at a position in the tensor
     * @param idx The index of the item. Can be flat or multidimensional.
     * @param dtype The dtype to return the value in
     */
    public <R> R item(DType<R> dtype, int... idx) {
        int flat = idx.length == 1 ? idx[0] : TensorUtils.flatIndex(this.shape.dimensions, idx);

        if (flat < 0 || flat > this.size())
            throw new IndexOutOfBoundsException("Invalid index for shape " + this.shape + ".");

        return dtype.parse(this.data[flat]);
    }

    /**
     * Set data at a multidimensional index of this Tensor
     * more information about indexing at `TensorUtils.index`
     *
     * @param idx The index. If it points to a direct element, the element will be set to obj.item(),
     *            if it points to a slice, the slice will be filled with obj
     */
    public Tensor<T> set(Tensor<T> obj, Integer... idx) { return TensorUtils.index_put(this, obj, idx); }

    /**
     * Set an object in the tensors data
     * @param obj The new object
     * @param idx The index of the object
     */
    public Tensor<T> set(T obj, int... idx) {
        boolean canPerform = idx.length == this.shape.rank();

        int flatIndex = TensorUtils.flatIndex(this.shape.dimensions, idx);

        canPerform = flatIndex < 0 || this.data.length >= flatIndex || canPerform;

        if (!canPerform)
            throw new IndexOutOfBoundsException("Invalid index for shape " + this.shape + " (index: " + Arrays.toString(idx) + ")");

        this.data[flatIndex] = obj;
        return this;
    }

    /**
     * Perform element wise addition between the values of this Tensor another value
     * @param other The other value
     */
    public Tensor<T> add(T other) { return this.add(Tensor.filled(other, this.shape.dimensions)); }

    /**
     * Perform element wise addition between the values of this Tensor and another one
     * @param other The second Tensor
     */
    public Tensor<T> add(Tensor<T> other) { return TensorUtils.elementWise(this, other, Double::sum); }

    /**
     * Perform element wise subtraction between the values of this Tensor another value
     * @param other The other value
     */
    public Tensor<T> sub(T other) { return this.sub(Tensor.filled(other, this.shape.dimensions)); }

    /**
     * Perform element wise subtraction between the values of this Tensor and another one
     * @param other The second Tensor
     */
    public Tensor<T> sub(Tensor<T> other) { return TensorUtils.elementWise(this, other, (a, b) -> a - b); }

    /**
     * Perform element wise division between the values of this Tensor another value
     * @param other The other value
     */
    public Tensor<T> div(T other) { return this.div(Tensor.filled(other, this.shape.dimensions)); }

    /**
     * Perform element wise division between the values of this Tensor and another one
     * @param other The second Tensor
     */
    public Tensor<T> div(Tensor<T> other) { return TensorUtils.elementWise(this, other, (a, b) -> a / b); }

    /**
     * Perform element wise multiplication between the values of this Tensor another value
     * @param other The other value
     */
    public Tensor<T> mul(T other) { return this.mul(Tensor.filled(other, this.shape.dimensions)); }

    /**
     * Perform element wise multiplication between the values of this Tensor and another one
     * @param other The second Tensor
     */
    public Tensor<T> mul(Tensor<T> other) { return TensorUtils.elementWise(this, other, (a, b) -> a * b); }

    /**
     * Raise each element to a power
     * @param power The power
     */
    public Tensor<T> pow(double power) { return this.pow(Tensor.filled(power, this.shape.dimensions)); }

    /**
     * Perform element wise power between the values of this Tensor and another one
     * @param power The second Tensor
     */
    public Tensor<T> pow(Tensor<?> power) { return TensorUtils.elementWise(this, power, Math::pow); }

    /**
     * Compute the square root for each element
     */
    public Tensor<T> sqrt() {
        return TensorUtils.elementWise(this, (a, i) -> Math.sqrt(DType.DOUBLE.parse(a)));
    }

    /**
     * Perform exponentiation on each element
     */
    public Tensor<T> exp() {
        return TensorUtils.elementWise(this, (a, i) -> Math.exp(DType.DOUBLE.parse(a)));
    }

    /**
     * Perform logarithm on each element
     */
    public Tensor<T> log() {
        return TensorUtils.elementWise(this, (a, i) -> Math.log(DType.DOUBLE.parse(a)));
    }

    /**
     * Clip each element at a min and max
     * @param min The lowest value an element can be
     * @param max The largest value an element can be
     */
    public Tensor<T> clip(double min, double max) { return TensorUtils.elementWise(this, (a, i) ->
                Math.max(Math.min(DType.DOUBLE.parse(a), max), min)); }

    /**
     * Get the largest element in the Tensors data
     */
    public T max() { return Arrays.stream(this.data).max((Comparator<? super T>) Comparator.naturalOrder()).orElseThrow(); }

    /**
     * Get the smallest element in the Tensors data
     */
    public T min() { return Arrays.stream(this.data).min((Comparator<? super T>) Comparator.naturalOrder()).orElseThrow(); }

    /**
     * Get the index of an element in the data array
     * @param obj The object to search for
     */
    public int indexOf(T obj) { return Arrays.stream(this.data).toList().indexOf(obj); }

    /**
     * Get the multidimensional index of an element in the data array
     * @param obj The object to search for
     */
    public int[] indexOf_md(T obj) { return TensorUtils.unravelIndex(this.shape.dimensions, this.indexOf(obj)); }

    /**
     * Reduce a specific dimension by summing its values
     * @param dim The dimension to sum over
     * @param keepDim Should the summed dimension be removed?
     *                More information at `TensorUtils.reduceAlongDimension`
     */
    public Tensor<T> sum(int dim, boolean keepDim) { return TensorUtils.reduceAlongDimension(this, dim, Tensor::add, keepDim); }

    /**
     * Reduce a specific dimension by averaging its values
     * @param dim The dimension to average over
     * @param keepDim Should the averaged dimension be removed?
     *                More information at `TensorUtils.reduceAlongDimension`
     */
    public Tensor<T> mean(int dim, boolean keepDim) { return this.sum(dim, keepDim).div(this.dtype.parse(this.size(dim))); }

    /**
     * Transposes the last two dimensions of the Tensor
     */
    public Tensor<T> T() { return this.transpose(-1, -2); }

    /**
     * Creates a transposed version of the Tensor where the dimensions `dim0` and `dim1` are swapped
     * @param dim0 The dimension to swap with `dim1`
     * @param dim1 The dimension to swap with `dim0`
     */
    public Tensor<T> transpose(int dim0, int dim1) {
        dim0 = DataUtils.handleNegativeIndexing(this.shape.dimensions, dim0);
        dim1 = DataUtils.handleNegativeIndexing(this.shape.dimensions, dim1);

        int[] newShape = this.shape.dimensions.clone();
        newShape[dim0] = this.size(dim1);
        newShape[dim1] = this.size(dim0);

        Tensor<T> result = this.clone().reshape(newShape);

        for (int i = 0; i < this.size(); i++) {
            int[] indices = TensorUtils.unravelIndex(this.shape.dimensions, i);
            int tempIndex = indices[dim0];
            indices[dim0] = indices[dim1];
            indices[dim1] = tempIndex;

            result.set(this.item(i), indices);
        }

        return result;
    }

    /**
     * Perform matrix multiplication over the last two dimensions of this tensor with another one
     * @param b The second Tensor
     */
    public Tensor<T> matmul(Tensor<T> b) {
        Tensor<T> a = this;

        boolean wasA1D = false, wasB1D = false;
        if (a.shape.rank() == 1) {
            a = a.unsqueeze(0);
            wasA1D = true;
        }
        if (b.shape.rank() == 1) {
            b = b.unsqueeze(0);
            wasB1D = true;
        }

        int aRows = a.size(-2);
        int aCols = a.size(-1);
        int bCols = b.size(-1);

        // pad a and b to same rank
        int length = Math.max(a.shape.rank(), b.shape.rank());
        a = a.reshape(TensorUtils.padShapeLeft(length, false, a.shape.dimensions));
        b = b.reshape(TensorUtils.padShapeLeft(length, false, b.shape.dimensions));

        // calculate broadcasted batch shape
        int[] batchShape = BroadcastingUtils.broadcastShapes(
                Arrays.copyOfRange(a.shape.dimensions, 0, a.shape.rank() - 2),
                Arrays.copyOfRange(b.shape.dimensions, 0, b.shape.rank() - 2)
        );
        int[] resultShape = Arrays.copyOfRange(batchShape, 0, batchShape.length + 2);
        resultShape[resultShape.length - 2] = aRows;
        resultShape[resultShape.length - 1] = bCols;
        Tensor<T> result = new Tensor<>(this.dtype, resultShape);

        // perform matrix multiplication
        for (int[] batchIndices : TensorUtils.calculatePossibleIndices(batchShape)) {
            Tensor<T> aSlice = a.get(DataUtils.IntegerIndex(batchIndices));
            Tensor<T> bSlice = b.get(DataUtils.IntegerIndex(batchIndices));
            Tensor<T> sliceResult = new Tensor<>(this.dtype, aRows, bCols);

            // perform matmul for this slice
            for (int i = 0; i < aRows; i++) {
                for (int j = 0; j < bCols; j++) {
                    // calculate dot product for result[..., i, j]
                    double sum = 0;
                    for (int k = 0; k < aCols; k++)
                        sum = sum + aSlice.item(DType.DOUBLE, i, k) * bSlice.item(DType.DOUBLE, k, j);

                    sliceResult.set(this.dtype.parse(sum), i, j);
                }
            }

            result.set(sliceResult, DataUtils.IntegerIndex(batchIndices));
        }

        if (wasA1D) result = result.squeeze(0);
        if (wasB1D) result = result.squeeze(-1);
        return result;
    }

    @Override
    public Tensor<T> clone() {
        return new Tensor<>(this.data.clone(), this.shape.dimensions.clone());
    }

    @Override
    public String toString() {
        return "Tensor{" +
                "shape=" + shape +
                ", size=" + shape.size() +
                ", dtype=" + dtype +
                ", data=" + Arrays.toString(data) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Tensor<?> && // same class
                Arrays.equals(((Tensor<?>) o).data, this.data) && // same data
                ((Tensor<?>) o).shape.equals(this.shape); // same shape
    }

    @Override
    public int hashCode() {
        return Objects.hash(Arrays.hashCode(data), shape, dtype);
    }
}