package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.BroadcastingUtils;

public class BroadcastOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected Integer[] shape;

    public BroadcastOperation(Tensor<T> a, Integer... shape) {
        super(a);

        this.a = this.inputs.getFirst();
        this.shape = shape;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        Integer[] broadcastedShape = BroadcastingUtils.broadcastShapes(this.a.shape.dimensions, shape);
        Tensor<T> result = this.a.clone().reshapeUnsafe(broadcastedShape);
        result.data = BroadcastingUtils.broadcastData(this.a.data, this.a.shape.dimensions, broadcastedShape);
        return result;
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[a -> $shape] = grad <- $original
        grad = grad.reduceToShape(this.getValue("aShape"));

        this.a.accumulate_grad(grad);
    }
}