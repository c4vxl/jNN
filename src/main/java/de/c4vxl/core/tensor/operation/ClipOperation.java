package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class ClipOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected double min, max;

    public ClipOperation(Tensor<T> a, double min, double max) {
        super(a);

        this.a = this.inputs.getFirst();
        this.min = min;
        this.max = max;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (a, i) ->
                Math.max(Math.min(DType.DOUBLE.parse(a), max), min));
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[clip(a, $min, $max)] = (a >= min) AND (input <= max) ? 1 : 0
        Tensor<T> mask = TensorUtils.elementWise(this.a, (val, i) -> {
            double v = DType.DOUBLE.parse(val);
            return (v >= min && v <= max) ? 1 : 0;
        });

        grad = grad.mul(mask);

        this.a.accumulate_grad(grad);
    }
}
