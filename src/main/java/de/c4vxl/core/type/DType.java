package de.c4vxl.core.type;

import java.util.Objects;
import java.util.Random;

public class DType<T> {
    // truth dtypes
    public static DType<Boolean> BOOLEAN = new DType<>(Boolean.class);

    // numerical dtypes
    public static DType<Integer> INTEGER = new DType<>(Integer.class);
    public static DType<Float> FLOAT = new DType<>(Float.class);
    public static DType<Long> LONG = new DType<>(Long.class);
    public static DType<Double> DOUBLE = new DType<>(Double.class);

    // default dtype
    public static DType<?> DEFAULT = DType.DOUBLE;

    public Class<T> clazz;
    public DType(Class<T> clazz) { this.clazz = clazz; }

    /**
     * Translate any kind of object into its representation in this DType
     * @param obj The object to translate
     */
    public T parse(Object obj) { return parse(this, obj); }

    /**
     * Translate any kind of object into its representation in another DType
     * @param obj The object to translate
     * @param dtype The target dtype
     */
    @SuppressWarnings("unchecked")
    public static <T> T parse(DType<T> dtype, Object obj) {
        if (obj == null) return null;
        if (dtype.clazz.isInstance(obj)) return (T) obj;

        // Try to convert obj into a Number
        Number number = obj instanceof Number ? (Number) obj : Double.valueOf(obj.toString());

        if (dtype.equals(INTEGER)) return (T) Integer.valueOf(number.intValue());
        if (dtype.equals(LONG)) return (T) Long.valueOf(number.longValue());
        if (dtype.equals(BOOLEAN)) return (T) Boolean.valueOf(number.intValue() > 0);
        if (dtype.equals(DOUBLE)) return (T) Double.valueOf(number.doubleValue());
        if (dtype.equals(FLOAT)) return (T) Float.valueOf(number.floatValue());
        if (obj instanceof Boolean bool) return dtype.parse(bool ? 1 : 0);

        // Avoid reflection unless absolutely necessary
        try {
            return (T) dtype.clazz.getDeclaredMethod("valueOf", String.class).invoke(null, obj.toString());
        } catch (ReflectiveOperationException e) {
            return null;
        }
    }

    /**
     * Returns a random value in this Dtype in between the bounds of 0 to 1, or 0 to 100 for Integers
     */
    public T randomValue() { return parse(new Random().nextDouble(0, this.equals(INTEGER) ? 100 : 1)); }

    /**
     * Returns a random value in this Dtype in between the bounds of `min` and `max`
     * @param min The minimal value
     * @param max The maximal value
     */
    public T randomValue(double min, double max) { return parse(new Random().nextDouble(min, max)); }

    @Override
    public String toString() {
        return clazz.getSimpleName();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DType<?> && ((DType<?>) o).clazz.equals(this.clazz);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(clazz);
    }
}
