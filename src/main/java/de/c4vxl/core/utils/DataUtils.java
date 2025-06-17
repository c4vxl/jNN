package de.c4vxl.core.utils;

import de.c4vxl.core.type.DType;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * A collection of utilities used handling general data.
 */
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
    public static int handleNegativeIndexing(Integer[] data, int idx) {
        if (idx < 0) return (data.length + (idx % data.length)) % data.length;
        return idx;
    }

    /**
     * Convert a negative multidimensional index into it's positive values
     * @param dimensions The dimensions to index into
     * @param idx The multidimensional index
     */
    public static Integer[] handleNegativeIndexing(Integer[] dimensions, Integer[] idx) {
        for (int i = 0; i < dimensions.length; i++)
            if (idx[i] != null && idx[i] < 0)
                idx[i] = (dimensions[i] + (idx[i] % dimensions[i])) % dimensions[i];
        return idx;
    }

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

    /**
     * Cut an array from the left side to a given target length
     * @param data The data to cut
     * @param targetLength The targeted length
     */
    public static <T> T[] cutLeft(T[] data, int targetLength) {
        return data.length > targetLength ? DataUtils.padLeft(data, null, targetLength, true) : data;
    }

    /**
     * Cut an array from the right side to a given target length
     * @param data The data to cut
     * @param targetLength The targeted length
     */
    public static <T> T[] cutRight(T[] data, int targetLength) {
        return data.length > targetLength ? DataUtils.padRight(data, null, targetLength, true) : data;
    }
}