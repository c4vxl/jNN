package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.TensorUtils;

import java.util.List;

public class MulOperation<T> extends Operation<T> {
    protected Tensor<T> a, b;

    public MulOperation(Tensor<T> a, Tensor<T> b) {
        super(a, b);

        this.a = this.inputs.get(0);
        this.b = this.inputs.get(1);

        this.saveForBackward("aShape", a.shape.dimensions);
        this.saveForBackward("bShape", b.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(a, b, (a, b) -> a * b);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> gradA = gradOutput.reduceToShape(this.getValue("aShape"));
        Tensor<T> gradB = gradOutput.reduceToShape(this.getValue("bShape"));

        // grad[a * b] = [ b, a ]
        gradA = gradA.mul(this.b.detach());
        gradB = gradB.mul(this.a.detach());

        this.a.accumulate_grad(gradA);
        this.b.accumulate_grad(gradB);
    }
}