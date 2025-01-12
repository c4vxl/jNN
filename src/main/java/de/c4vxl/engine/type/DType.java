package de.c4vxl.engine.type;

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
        if (obj == null) return null; // null won't change
        if (dtype.clazz.isInstance(obj)) return (T) obj; // return if obj is already target DType

        // round to next full integer if target is DType.INTEGER
        if (dtype.equals(INTEGER))
            return (T) Integer.valueOf((int) Math.round(DType.DOUBLE.parse(obj)));

        // round to next full integer if target is DType.LONG
        if (dtype.equals(LONG))
            return (T) Long.valueOf(DType.INTEGER.parse(obj));

        // check if int representation is larger than 0 in case of DType.BOOLEAN
        if (dtype.equals(BOOLEAN))
            return (T) (DType.INTEGER.parse(obj) > 0 ? Boolean.TRUE : Boolean.FALSE);

        // if obj is BOOLEAN: if obj is True, return representation of 1; if obj is False, return 0;
        if (obj == Boolean.TRUE) return dtype.parse(1);
        if (obj == Boolean.FALSE) return dtype.parse(0);

        try {
            return (T) dtype.clazz.getMethod("valueOf", String.class).invoke(null, obj.toString());
        } catch (Exception e) {
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
