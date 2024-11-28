package de.c4vxl.engine.data;

public class DType {
    public static Class<Double> DEFAULT = Double.class;

    public static Class<Double> DOUBLE = Double.class;
    public static Class<Float> FLOAT = Float.class;
    public static Class<Long> LONG = Long.class;
    public static Class<Integer> INTEGER = Integer.class;
    public static Class<Boolean> BOOLEAN = Boolean.class;

    /**
     * Get the representation of a value in a specified data type
     * @param dtype Specify the data type
     * @param val value to parse
     */
    @SuppressWarnings("unchecked")
    public static <T> T valueOf(Class<?> dtype, Object val) {
        String v = val.toString();

        // no comma for Integers and Booleans
        if (dtype == Integer.class || dtype == Boolean.class)
            v = v.split("\\.")[0];

        // need to handle boolean logic seperate
        if (dtype == Boolean.class) {
            if (v.equals("1")) return (T) Boolean.TRUE;
            else return (T) Boolean.FALSE;
        }

        try {
            return (T) dtype.getMethod("valueOf", String.class).invoke(null, v);
        } catch (Exception e) {
            return null;
        }
    }
}
