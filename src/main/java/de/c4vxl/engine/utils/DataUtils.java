package de.c4vxl.engine.utils;

import de.c4vxl.engine.type.DType;

import java.lang.reflect.Array;
import java.util.Arrays;

public class DataUtils {
    /**
     * Fills the passed `data`-array with random values between 0...1
     * @param data The data to randomize
     */
    @SuppressWarnings("unchecked")
    public static <T> void randomInitialization(T[] data) {
        DType<T> dtype = (DType<T>) new DType<>(data.getClass().getComponentType());
        for (int i = 0; i < data.length; i++)
            data[i] = dtype.randomValue(0, 1);
    }

    /**
     * Convert a negative index into it's positive value
     * @param data The data to index into
     * @param idx The index into the dimensions
     */
    public static int handleNegativeIndexing(int[] data, int idx) { return idx < 0 ? data.length + idx : idx; }

    /**
     * Convert a negative multidimensional index into it's positive values
     * @param dimensions The dimensions to index into
     * @param idx The multidimensional index
     */
    public static Integer[] handleNegativeIndexing(int[] dimensions, Integer[] idx) {
        for (int i = 0; i < dimensions.length; i++)
            if (idx[i] != null)
                idx[i] = idx[i] >= 0 ? idx[i] : dimensions[i] + idx[i];
        return idx;
    }

    /**
     * Convert an index in Integer format into an int-index
     * @param idx The index
     */
    public static int[] intIndex(Integer... idx) { return Arrays.stream(idx).mapToInt(Integer::intValue).toArray(); }

    /**
     * Convert an index in int format into an Integer-index
     * @param idx The index
     */
    public static Integer[] IntegerIndex(int... idx) { return Arrays.stream(idx).boxed().toArray(Integer[]::new); }

    /**
     * Pad a Data array to fit a given size
     * @param data The array to pad
     * @param padWith The object to pad the data with
     * @param targetLength The targeted length for the data
     * @param cut Defines if the data should be cut if it already was longer than `targetLength`
     */
    public static <T> T[] padRight(T[] data, T padWith, int targetLength, boolean cut) {
        if (!cut && data.length > targetLength)
            return data;

        T[] result = Arrays.copyOfRange(data, 0, targetLength);

        for (int i = 0; i < result.length; i++)
            result[i] = data.length > i ? data[i] : padWith;

        return result;
    }

    /**
     * Pad a Data array to fit a given size
     * @param data The array to pad
     * @param padWith The object to pad the data with
     * @param targetLength The targeted length for the data
     * @param cut Defines if the data should be cut if it already was longer than `targetLength`
     */
    @SuppressWarnings("unchecked")
    public static <T> T[] padLeft(T[] data, T padWith, int targetLength, boolean cut) {
        int offset = targetLength - data.length;

        if (!cut && offset < 0)
            return data;

        T[] result = (T[]) Array.newInstance(data.getClass().getComponentType(), targetLength);
        for (int i = 0; i < targetLength; i++)
            result[i] = i < offset ? padWith : data[i - offset];

        return result;
    }
}
