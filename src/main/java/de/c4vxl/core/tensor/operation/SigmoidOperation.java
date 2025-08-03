package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class SigmoidOperation<T> extends Operation<T> {
    protected Tensor<T> a;

    public SigmoidOperation(Tensor<T> a) {
        super(a);

        this.a = this.inputs.getFirst();

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (a, i) -> 1 / (1 + Math.exp(-1 * DType.DOUBLE.parse(a))));
    }

    @SuppressWarnings("unchecked")
    @Override
    public void _backward(Tensor<T> gradOutput) {
        // σ(x)⋅(1−σ(x))
        Tensor<T> sigA = new SigmoidOperation<>(this.a.detach()).forward();
        Tensor<T> grad = sigA.clone().mul(Tensor.of(this.a.dtype.parse(1)).sub(sigA));
        grad = gradOutput.mul(grad);

        this.a.accumulate_grad(grad);
    }
}