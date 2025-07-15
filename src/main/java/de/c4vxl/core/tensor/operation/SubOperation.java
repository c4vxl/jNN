package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.TensorUtils;

public class SubOperation<T> extends Operation<T> {
    protected Tensor<T> a, b;

    public SubOperation(Tensor<T> a, Tensor<T> b) {
        super(a, b);

        this.a = this.inputs.get(0);
        this.b = this.inputs.get(1);

        this.saveForBackward("aShape", a.shape.dimensions);
        this.saveForBackward("bShape", b.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(a, b, (a, b) -> a - b);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> gradA = gradOutput.reduceToShape(this.getValue("aShape"));
        Tensor<T> gradB = gradOutput.reduceToShape(this.getValue("bShape"));

        // grad[a - b] = [ 1, -1 ] --> negate b

        this.a.accumulate_grad(gradA);
        this.b.accumulate_grad(gradB.neg());
    }
}