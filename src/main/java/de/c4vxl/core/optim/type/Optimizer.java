package de.c4vxl.core.optim.type;

public interface Optimizer {
    void step();
    void zeroGrad();
}
