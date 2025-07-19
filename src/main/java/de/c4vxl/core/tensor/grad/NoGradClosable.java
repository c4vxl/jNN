package de.c4vxl.core.tensor.grad;

public class NoGradClosable implements AutoCloseable {
    private final boolean prevState;

    public NoGradClosable() {
        this.prevState = GradContext.isNoGrad();
        GradContext.setNoGrad(true);
    }

    @Override
    public void close() {
        GradContext.setNoGrad(prevState);
    }
}