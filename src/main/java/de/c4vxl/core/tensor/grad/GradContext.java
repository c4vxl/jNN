package de.c4vxl.core.tensor.grad;

import de.c4vxl.jNN;

public class GradContext {
    private static final ThreadLocal<Boolean> noGrad = ThreadLocal.withInitial(() -> !jNN.DEFAULT_REQUIRE_GRADIENT);

    /**
     * Checks if gradient is enabled in the current thread
     */
    public static boolean isNoGrad() { return noGrad.get(); }

    /**
     * Set the no-grad context
     * @param value Whether nograd should be enabled or not
     */
    public static void setNoGrad(boolean value) {
        noGrad.set(value);
    }

    /**
     * Run a method with nograd enabled
     * @param block The method to run
     */
    public static void noGrad(Runnable block) {
        boolean prev = isNoGrad();
        setNoGrad(true);

        try {
            block.run();
        } finally {
            setNoGrad(prev);
        }
    }

    /**
     * Returns an AutoClosable instance that can be used in a {@code try}-block
     * to execute code without calculating gradients.
     */
    public static NoGradClosable noGrad() { return new NoGradClosable(); }
}