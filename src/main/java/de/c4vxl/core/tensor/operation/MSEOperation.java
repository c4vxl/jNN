package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;

public class MSEOperation<T> extends Operation<T> {
    protected Tensor<T> output, target;

    public MSEOperation(Tensor<T> output, Tensor<T> target) {
        super(output, target);

        this.output = this.inputs.get(0);
        this.target = this.inputs.get(1);
    }

    @Override
    public Tensor<T> _forward() {
        // Copy out to local state
        // This is done so that reshapes applied by backpropagation (in .sub) won't propagate to the backward pass
        Tensor<T> out = this.output.clone();

        return out.sub(this.target).pow(2.).sum(0, true);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> diff = this.output.detach().sub(this.target.detach());
        Tensor<T> grad = diff.mul(diff.dtype.parse(2)).mul(gradOutput);

        this.output.accumulate_grad(grad);
    }
}