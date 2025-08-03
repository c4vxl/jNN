package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class ExpOperation<T> extends Operation<T> {
    protected Tensor<T> a;

    public ExpOperation(Tensor<T> a) {
        super(a);

        this.a = this.inputs.getFirst();

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (a, i) -> Math.exp(DType.DOUBLE.parse(a)));
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> mask = this.a.detach().exp();
        Tensor<T> grad = gradOutput.mul(mask);

        this.a.accumulate_grad(grad);
    }
}