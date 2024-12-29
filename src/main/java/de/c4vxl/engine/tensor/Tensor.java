package de.c4vxl.engine.tensor;

import de.c4vxl.engine.type.DType;
import de.c4vxl.engine.type.Shape;
import de.c4vxl.engine.utils.DataUtils;
import de.c4vxl.engine.utils.TensorUtils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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
            return Tensor.of(this.data[TensorUtils.flatIndex(this.shape.dimensions, DataUtils.intIndex(idx))]);

        // points to a slice
        return TensorUtils.index(this, idx);
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
     *
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