package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.type.Shape;

import java.lang.reflect.Array;

public class ReshapeOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected Integer[] newShape;

    public ReshapeOperation(Tensor<T> a, Integer... newShape) {
        super(a);

        this.a = this.inputs.getFirst();
        this.newShape = newShape;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Tensor<T> _forward() {
        Tensor<T> result = this.a.clone();
        result.shape = new Shape(newShape);

        // Return if booth elements are the same size
        if (result.size() == this.a.size()) return result;

        result.data = (T[]) Array.newInstance(this.a.dtype.clazz, result.shape.size());
        for (int i = 0; i < result.data.length; i++)
            result.data[i] = this.a.data.length > i ? this.a.data[i] : null;

        return result;
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[reshape(a, $shape)] = out.reshape($original)
        grad = grad.reshape(this.getValue("aShape"));

        this.a.accumulate_grad(grad);
    }
}