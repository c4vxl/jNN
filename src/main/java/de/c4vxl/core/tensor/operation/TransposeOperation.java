package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.DataUtils;
import de.c4vxl.core.utils.TensorUtils;

public class TransposeOperation<T> extends Operation<T> {
    protected Tensor<T> a;
    protected int dim0, dim1;

    public TransposeOperation(Tensor<T> a, int dim0, int dim1) {
        super(a);

        this.a = this.inputs.getFirst();
        this.dim0 = dim0;
        this.dim1 = dim1;

        this.saveForBackward("aShape", a.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        dim0 = DataUtils.handleNegativeIndexing(this.a.shape.dimensions, this.dim0);
        dim1 = DataUtils.handleNegativeIndexing(this.a.shape.dimensions, this.dim1);

        Integer[] newShape = this.a.shape.dimensions.clone();
        newShape[dim0] = this.a.size(dim1);
        newShape[dim1] = this.a.size(dim0);

        Tensor<T> result = this.a.clone().reshape(newShape);

        for (int i = 0; i < this.a.size(); i++) {
            Integer[] indices = TensorUtils.unravelIndex(this.a.shape.dimensions, i);
            int tempIndex = indices[dim0];
            indices[dim0] = indices[dim1];
            indices[dim1] = tempIndex;

            result.set(this.a.data[i], indices);
        }

        return result;
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> grad = gradOutput.reduceToShape(this.getValue("aShape"));

        // grad[transpose(a)] = out.transpose(dim0, dim1)
        grad = grad.transpose(dim0, dim1);

        this.a.accumulate_grad(grad);
    }
}