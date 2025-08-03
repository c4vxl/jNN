package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class LeakyReLUOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected double alpha;

    public LeakyReLUOperation(Tensor<T> a, Double alpha) {
        super(a);

        this.a = this.inputs.getFirst();
        this.alpha = alpha;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (x, ignored) -> {
            Double xVal = DType.DOUBLE.parse(x);
            return xVal > 0 ? xVal : xVal * this.alpha;
        });
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[LeakyReLU(a)] = (a > 1) ? 1 : 0
        Tensor<T> mask = TensorUtils.elementWise(this.a, (val, i) -> {
            double v = DType.DOUBLE.parse(val);
            return v > 0 ? 1 : this.alpha;
        });

        grad = grad.mul(mask);

        this.a.accumulate_grad(grad);
    }
}