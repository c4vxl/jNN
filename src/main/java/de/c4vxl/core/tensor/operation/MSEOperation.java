package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;

public class MSEOperation<T> extends Operation<T> {
    protected Tensor<T> output, target;

    public String reduction;

    public MSEOperation(Tensor<T> output, Tensor<T> target) { this(output, target, "mean"); }

    public MSEOperation(Tensor<T> output, Tensor<T> target, String reduction) {
        super(output, target);

        this.output = this.inputs.get(0);
        this.target = this.inputs.get(1);
        this.reduction = reduction;
    }

    @Override
    public Tensor<T> _forward() {
        // Copy out to local state
        // This is done so that reshapes applied by backpropagation (in .sub) won't propagate to the backward pass
        Tensor<T> out = this.output.clone();
        Tensor<T> b = out.sub(this.target).pow(2.);

        // Handle different reductions
        if (reduction.equalsIgnoreCase("mean"))
            return b.mean(0, true);

        return b.sum(0, true);
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> diff = this.output.detach().sub(this.target.detach());
        Tensor<T> grad = diff.mul(diff.dtype.parse(2)).mul(gradOutput);

        this.output.accumulate_grad(grad);
    }
}