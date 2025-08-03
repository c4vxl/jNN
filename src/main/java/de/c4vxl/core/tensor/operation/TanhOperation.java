package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class TanhOperation<T> extends Operation<T> {
    protected Tensor<T> a;

    public TanhOperation(Tensor<T> a) {
        super(a);

        this.a = this.inputs.getFirst();

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (a, i) -> Math.tanh(DType.DOUBLE.parse(a)));
    }

    @SuppressWarnings("unchecked")
    @Override
    public void _backward(Tensor<T> gradOutput) {
        // 1 - tanh(x)²
        Tensor<T> grad = Tensor.of(this.a.dtype.parse(1)).sub(this.a.detach().tanh().pow(2));
        grad = gradOutput.mul(grad);

        this.a.accumulate_grad(grad);
    }
}