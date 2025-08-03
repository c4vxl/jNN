package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.nn.activation.type.ActivationFunction;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;

public class CrossEntropyLossOperation<T> extends Operation<T> {
    protected Tensor<T> output, target;

    public CrossEntropyLossOperation(Tensor<T> output, Tensor<T> target) {
        super(output, target);

        this.output = this.inputs.get(0);
        this.target = this.inputs.get(1);
    }

    @Override
    public Tensor<T> _forward() {
        // Copy out to local state
        // This is done so that changes applied to them won't propagate to the backward pass
        Tensor<T> out = this.output, target = this.target;
        out = ActivationFunction.Softmax(out);

        // Clip to avoid log(0)
        out = out.clip(1e-7, 1 - 1e-7);

        return target.mul(out.log()).mul(target.dtype.parse(-1.)).sum(-1, true);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> out = ActivationFunction.Softmax(this.output.detach());
        Tensor<T> grad = out.sub(this.target.detach()).mul(gradOutput);

        this.output.accumulate_grad(grad);
    }
}