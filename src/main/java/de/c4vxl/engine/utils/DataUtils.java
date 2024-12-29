package de.c4vxl.engine.utils;

import de.c4vxl.engine.type.DType;

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
}
