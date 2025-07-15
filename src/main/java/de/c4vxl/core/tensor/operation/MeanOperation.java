package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;

public class MeanOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected int dim;
    protected boolean keepDim;

    public MeanOperation(Tensor<T> a, int dim, boolean keepDim) {
        super(a);

        this.a = this.inputs.getFirst();
        this.dim = dim;
        this.keepDim = keepDim;

        this.saveForBackward("aShape", a.shape.dimensions);
        this.saveForBackward("size", a.size(dim));
    }

    @Override
    public Tensor<T> _forward() { return this.a.sum(this.dim, this.keepDim).div(this.a.dtype.parse(this.a.size(dim))); }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[mean(a)] = (a -> originalShape) / size($dim)
        grad = grad.broadcastTo((Integer[]) this.getValue("aShape")).div(this.a.dtype.parse(this.getValue("size")));

        this.a.accumulate_grad(grad);
    }
}