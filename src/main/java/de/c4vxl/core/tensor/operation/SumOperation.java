package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.TensorUtils;

public class SumOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected int dim;
    protected boolean keepDim;

    public SumOperation(Tensor<T> a, int dim, boolean keepDim) {
        super(a);

        this.a = this.inputs.getFirst();
        this.dim = dim;
        this.keepDim = keepDim;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() { return TensorUtils.reduceAlongDimension(this.a, this.dim, Tensor::add, this.keepDim); }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[sum(a)] = a -> originalShape
        grad = grad.broadcastTo((Integer[]) this.getValue("aShape"));

        this.a.accumulate_grad(grad);
    }
}