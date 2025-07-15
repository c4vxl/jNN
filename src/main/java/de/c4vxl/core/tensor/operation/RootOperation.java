package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class RootOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected double degree;

    public RootOperation(Tensor<T> a, double degree) {
        super(a);

        this.a = this.inputs.getFirst();
        this.degree = degree;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        // Use Math.sqrt for degree = 2
        if (this.degree == 2.)
            return TensorUtils.elementWise(this.a, (a, i) -> Math.sqrt(DType.DOUBLE.parse(a)));

        return this.a.pow(1. / this.degree);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        double exponent = (1.0 / this.degree) - 1.0;
        Tensor<T> pow = this.a.pow(exponent);

        Tensor<T> factor = pow.mul(this.a.dtype.parse(1.0 / this.degree));
        Tensor<T> result = grad.mul(factor);

        this.a.accumulate_grad(result);
    }
}