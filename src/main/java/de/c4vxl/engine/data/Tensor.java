package de.c4vxl.engine.data;

import java.lang.reflect.Array;
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
    // default data type
    public static Class<?> defaultDataType = Float.class;

    public boolean is2d() { return this.shape.length == 2; }
    public boolean is3d() { return this.shape.length == 3; }

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
        return new Tensor<>((Class<T>) obj.getClass(), shape).fill(obj);
    }

    /**
     * Construct a Tensor filled with zeros
     * @param shape Shape of the Tensor
     */
    public static Tensor<Integer> zeros(int... shape) { return Tensor.of(0, shape); }

    /**
     * Construct a Tensor filled with zeros
     * @param dtype Datatype of the Tensor
     * @param shape Shape of the Tensor
     */
    public static <T> Tensor<T> zeros(Class<T> dtype, int... shape) { return Tensor.of(Objects.requireNonNull(valueOf(dtype, "0")), shape); }

    /**
     * Construct a Tensor filled with ones
     * @param shape Shape of the Tensor
     */
    public static Tensor<Integer> ones(int... shape) { return Tensor.of(1, shape); }

    /**
     * Construct a Tensor filled with ones
     * @param dtype Datatype of the Tensor
     * @param shape Shape of the Tensor
     */
    public static <T> Tensor<T> ones(Class<T> dtype, int... shape) { return Tensor.of(Objects.requireNonNull(valueOf(dtype, "1")), shape); }

    /**
     * Construct a Tensor filled with numbers of a range
     * @param start The start of the range
     * @param end The end of the range
     * @param stepSize The size of the steps between start and end
     */
    public static <T> Tensor<T> range(Class<T> dtype, int start, int end, int stepSize) {
        if (start > end)
            throw new IllegalArgumentException("Start point can not be larger than end point!");

        int size = (end - start) / stepSize + 1;

        Object[] data = (Object[]) Array.newInstance(dtype, size);

        for (int i = 0; i < size; i++)
            data[i] = valueOf(dtype, "" + (start + i * stepSize));

        return new Tensor<>((T[]) data, 1, size);
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
     * Get the representation of a value in a specified data type
     * @param dtype Specify the data type
     * @param val value to parse
     */
    public static <T> T valueOf(Class<?> dtype, String val) {
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
     * Get the representation of a value in current dtype
     * @param val Your value to parse to the current dtype
     */
    public T valueOf(String val) { return valueOf(dtype, val); }

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
     * Perform matrix multiplication with another Tensor
     * @param b The other Tensor to multiply with
     */
    public Tensor<T> matmul(Tensor<T> b) {
        if (!is2d() || !b.is2d())
            throw new IllegalArgumentException("Matmul is only supported for 2d tensors!");

        if (this.shape[1] != b.shape[0])
            throw new IllegalArgumentException("Incompatible shapes for matrix multiplication: " +
                    Arrays.toString(this.shape) + " and " + Arrays.toString(b.shape));

        int m = this.shape[0];
        int n = this.shape[1];
        int p = b.shape[1];
        T[] resultData = (T[]) Array.newInstance(dtype, m * p);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                T sum = valueOf("0");
                for (int k = 0; k < n; k++) {
                    T a = this.data[i * n + k];
                    T o = b.data[k * p + j];

                    sum = numericalOperation(sum, numericalOperation(a, o, (t, c) -> t * c), Double::sum);
                }
                resultData[i * p + j] = sum;
            }
        }

        return new Tensor<>(resultData, m, p);
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
        for (int i = 0; i < dims.length; i++)
            dims[i] = dims[i] < 0 ? shape.length + dims[i] : dims[i];

        // calculate new shape and new data array
        int[] newShape = Arrays.stream(dims).map(d -> shape[d]).toArray();
        T[] resultData = (T[]) Array.newInstance(data.getClass().getComponentType(),
                Arrays.stream(newShape).reduce(1, (a, b) -> a * b));

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