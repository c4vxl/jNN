package de.c4vxl;

public class jNN {
    /**
     * The name of the current version
     */
    public static final String VERSION = "1.0.0";

    /**
     * If set to true, errors in state generation will be logged.
     */
    public static boolean LOG_STATE_GENERATION_ERROR = false;

    /**
     * The type of matrix multiplication.
     * If set to 1 "nd4j" will be used; If set to 0 own implementation will be used!
     */
    public static int MATMUL_TYPE = 1;
}
