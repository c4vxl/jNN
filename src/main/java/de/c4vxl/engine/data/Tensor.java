package de.c4vxl.engine.data;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;

@SuppressWarnings("unchecked")
public class Tensor<T> {
    // default data type
    public static Class<?> defaultDataType = Float.class;

    public Class<T> dtype;
    public T[] data;
    public int[] shape;
    public int size;

    /**
     * Construct a Tensor with the default datatype
     * @param shape The shape of the Tensor
     */
    public Tensor(int... shape) { this((Class<T>) defaultDataType, shape); }

    /**
     * Construct a Tensor with a specified datatype
     * @param dtype Data Type of the Tensor
     * @param shape The shape of the Tensor
     */
    public Tensor(Class<T> dtype, int... shape) {
        this((T[]) Array.newInstance(dtype, 1), shape);

        Random rand = new Random();
        for (int i = 0; i < data.length; i++) {
            if (dtype == Double.class) data[i] = (T) Double.valueOf(rand.nextDouble());
            else if (dtype == Integer.class) data[i] = (T) Integer.valueOf(rand.nextInt(99));
            else if (dtype == Long.class) data[i] = (T) Long.valueOf(rand.nextLong());
            else if (dtype == Float.class) data[i] = (T) Float.valueOf(rand.nextFloat());
            else if (dtype == Boolean.class) data[i] = (T) Boolean.valueOf(rand.nextBoolean());
            else throw new IllegalArgumentException("Unsupported dtype '" + dtype.getSimpleName() + "'");
        }
    }

    /**
     * Construct a Tensor with preconfigured data
     * @param data Data of the Tensor
     * @param shape Shape of the Tensor
     */
    public Tensor(T[] data, int... shape) {
        this.shape = shape;
        this.size = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
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
    public static <T> Tensor<T> of(T obj, int... shape) {
        return (Tensor<T>) new Tensor<>(new Object[]{obj}, shape);
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
    public T valueOf(String val) {
        // no comma for Integers and Booleans
        if (dtype == Integer.class || dtype == Boolean.class)
            val = val.split("\\.")[0];

        try {
            return (T) dtype.getMethod("valueOf", String.class).invoke(null, val);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Flatten the index of the location in the Tensor
     * @param idx Location in the Tensor
     */
    public int flatIndex(int... idx) {
        int flatIndex = 0;
        for (int i = 0; i < idx.length; i++)
            flatIndex = flatIndex * shape[i] + idx[i];

        return flatIndex;
    }

    /**
     * Get the item in this Tensor
     * @param loc Location of the item, if not passed, the first item will be used
     */
    public T item(int... loc) {
        return data[loc.length > 0 ? flatIndex(loc) : 0];
    }

    /**
     * Perform an operation on each element in this Tensor
     */
    public Tensor<T> elementWise(BiFunction<T, Integer, Object> task) {
        Tensor<T> out = this.clone();
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (T) task.apply(this.data[i], i);
        }

        return out;
    }

    /**
     * Perform a numerical operation between two values with unknown dtype
     */
    public T numericalOperation(T a, T b, BiFunction<Double, Double, Double> operation) {
        if (dtype == Boolean.class)
            throw new RuntimeException("Operation can not be performed on dtype 'Boolean'");

        Double result = operation.apply(((Number) a).doubleValue(), ((Number) b).doubleValue());
        return valueOf(result.toString());
    }

    /**
     * Perform element wise addition
     * @param other Pass the other tensor
     */
    public Tensor<T> add(Tensor<T> other) {
        return this.clone().elementWise((a, i) ->
                numericalOperation(a, other.data[i], Double::sum)
        );
    }

    /**
     * Perform element wise subtraction
     * @param other Pass the other tensor
     */
    public Tensor<T> sub(Tensor<T> other) {
        return this.clone().elementWise((a, i) ->
                numericalOperation(a, other.data[i], (o, s) -> o - s)
        );
    }

    /**
     * Perform element wise division
     * @param other Pass the other tensor
     */
    public Tensor<T> div(Tensor<T> other) {
        return this.clone().elementWise((a, i) ->
                numericalOperation(a, other.data[i], (o, s) -> o / s)
        );
    }

    /**
     * Perform element wise multiplication
     * @param other Pass the other tensor
     */
    public Tensor<T> mul(Tensor<T> other) {
        return this.clone().elementWise((a, i) ->
                numericalOperation(a, other.data[i], (o, s) -> o * s)
        );
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
     * @param shape New shape of the Tensor. Important: Final size must still be the same!
     */
    public Tensor<T> reshape(int... shape) {
        if (Arrays.stream(shape).reduce(1, (a, b) -> a * b) != this.size)
            throw new IllegalArgumentException("New shape must still be the same size as the old one!");

        Tensor<T> res = this.clone();
        res.shape = shape;
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
     * Sum across a given axis
     * @param axis The axis to sum over
     */
    public Tensor<T> sum(int axis) {
        // TODO: Implement sum
        return null;
    }

    /**
     * Perform matrix multiplication with another Tensor
     * @param b The other Tensor to multiply with
     */
    public Tensor<T> matmul(Tensor<T> b) {
        // TODO: Implement matrix multiplication
        return null;
    }

    /**
     * Transpose the Tensor's data over different dimensions
     * @param dims The dimensions to transpose the Tensor over
     */
    public Tensor<T> transpose(int... dims) {
        // TODO: Implement transposing
        return null;
    }

    @Override
    protected Tensor<T> clone() {
        return new Tensor<>(data.clone(), shape.clone());
    }

    @Override
    public String toString() {
        return "Tensor{" +
                "dtype=" + dtype.getSimpleName() +
                ", shape=" + Arrays.toString(shape) +
                ", size=" + size +
                ", data=" + Arrays.toString(data) +
                '}';
    }
}