package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.DType;
import de.c4vxl.core.utils.TensorUtils;

public class GELUOperation<T> extends Operation<T> {
    protected Tensor<T> a;

    public GELUOperation(Tensor<T> a) {
        super(a);

        this.a = this.inputs.getFirst();

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        return TensorUtils.elementWise(this.a, (a, i) -> {
            double x = DType.DOUBLE.parse(a);
            return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
        });
    }

    @SuppressWarnings("unchecked")
    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> x = this.a.detach();

        // sqrt(2 / pi)
        Tensor<T> sqrt2OverPi = Tensor.of(2.).div(Math.PI).sqrt().asDType(x.dtype);

        // s = sqrt(2 / pi) * (x + 0.044715 * x³)
        Tensor<T> s = sqrt2OverPi.mul(x.add(x.pow(3.).mul(x.dtype.parse(0.044715))));

        Tensor<T> tanhS = s.tanh();

        // sech²(s) = 1 - tanh²(s)
        Tensor<T> sech2 = Tensor.of(1.).asDType(x.dtype).sub(tanhS.pow(2.));

        // term1 = 0.5 * tanh(s)
        Tensor<T> term1 = tanhS.mul(x.dtype.parse(0.5));

        // term2 = 0.5 * x * sech²(s) * sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
        Tensor<T> term2 = x.mul(x.dtype.parse(0.5))
                .mul(sech2)
                .mul(sqrt2OverPi)
                .mul(Tensor.of(x.dtype.parse(1)).add(x.pow(2.).mul(x.dtype.parse(3 * 0.044715))).asDType(x.dtype));

        Tensor<T> gradInput = term1.add(term2).add(x.dtype.parse(0.5)); // term1 + term2 + 0.5

        gradInput = gradInput.mul(gradOutput);

        this.a.accumulate_grad(gradInput);
    }
}