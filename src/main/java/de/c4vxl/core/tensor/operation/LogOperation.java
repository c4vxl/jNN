package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class LogOperation<T> extends Operation<T> {
    protected Tensor<T> a;

    public LogOperation(Tensor<T> a) {
        super(a);

        this.a = this.inputs.getFirst();

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(a, (a, i) -> Math.log(DType.DOUBLE.parse(a)));
    }

    @SuppressWarnings("unchecked")
    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[log(a)] = a⁻¹
        grad = grad.div(a.detach().max((Tensor<T>) a.dtype.parse(1e-7)));  // using max(a, 1e-7) as log(a) with a <= 0 would be undefined

        this.a.accumulate_grad(grad);
    }
}