package de.c4vxl.engine.tensor;

import de.c4vxl.engine.type.DType;
import de.c4vxl.engine.type.Shape;
import de.c4vxl.engine.utils.DataUtils;
import de.c4vxl.engine.utils.TensorUtils;

import java.lang.reflect.Array;
import java.util.Arrays;

@SuppressWarnings("unchecked")
public class Tensor<T> {
    public T[] data;
    public Shape shape;
    public DType<T> dtype;

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
    public static Tensor<?> empty(int... shape) { return Tensor.filled(new Object[] { null }, shape); }

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
}