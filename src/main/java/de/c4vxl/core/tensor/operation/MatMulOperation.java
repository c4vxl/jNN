package de.c4vxl.core.tensor.operation;

import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.tensor.operation.type.Operation;
import de.c4vxl.core.utils.BroadcastingUtils;
import de.c4vxl.core.utils.TensorUtils;
import de.c4vxl.jNN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class MatMulOperation<T> extends Operation<T> {
    protected Tensor<T> a, b;

    public MatMulOperation(Tensor<T> a, Tensor<T> b) {
        super(a, b);

        this.a = this.inputs.get(0);
        this.b = this.inputs.get(1);

        this.saveForBackward("aShape", a.shape.dimensions);
        this.saveForBackward("bShape", b.shape.dimensions);
    }

    @Override
    public Tensor<T> _forward() {
        boolean wasA1D = false, wasB1D = false;
        if (a.shape.rank() == 1) {
            a = a.unsqueeze(0);
            wasA1D = true;
        }
        if (b.shape.rank() == 1) {
            b = b.unsqueeze(0);
            wasB1D = true;
        }

        // pad a and b to same rank
        int length = Math.max(a.shape.rank(), b.shape.rank());
        a = a.reshape(TensorUtils.padShapeLeft(length, false, a.shape.dimensions));
        b = b.reshape(TensorUtils.padShapeLeft(length, false, b.shape.dimensions));

        Tensor<T> result;

        if (jNN.MATMUL_TYPE == 1) { // nd4j version
            INDArray ndarray = Nd4j.matmul(Nd4j.createFromArray(a.asDouble().data).reshape(Arrays.stream(a.shape.dimensions).mapToInt(Integer::intValue).toArray()),
                    Nd4j.createFromArray(b.asDouble().data).reshape(Arrays.stream(b.shape.dimensions).mapToInt(Integer::intValue).toArray()));

            result = new Tensor<>(
                    Arrays.stream(ndarray.data().asDouble()).boxed().toArray(Double[]::new),
                    Arrays.stream(ndarray.shape()).boxed().map(Long::intValue).toArray(Integer[]::new)
            ).asDType(a.dtype);
        } else {                    // own version
            int aRows = a.size(-2);
            int aCols = a.size(-1);
            int bCols = b.size(-1);

            // calculate broadcasted batch shape
            Integer[] batchShape = BroadcastingUtils.broadcastShapes(
                    Arrays.copyOfRange(a.shape.dimensions, 0, a.shape.rank() - 2),
                    Arrays.copyOfRange(b.shape.dimensions, 0, b.shape.rank() - 2)
            );
            Integer[] resultShape = Arrays.copyOfRange(batchShape, 0, batchShape.length + 2);
            resultShape[resultShape.length - 2] = aRows;
            resultShape[resultShape.length - 1] = bCols;
            result = new Tensor<>(a.dtype, resultShape);

            // perform matrix multiplication
            TensorUtils.performBlockMultiplication(a, b, result, aRows, aCols, bCols,
                    0, 0, 0, 0, 32);
        }

        if (wasA1D) result = result.squeeze(0);
        if (wasB1D) result = result.squeeze(-1);

        return result;
    }

    @Override
    public void _backward(Tensor<T> gradOutput) {
        Tensor<T> gradA = gradOutput.reduceToShape(this.getValue("aShape"));
        Tensor<T> gradB = gradOutput.reduceToShape(this.getValue("bShape"));

        // grad[a @ b] = [ b.T(), a.T() ]

        gradA = gradA.matmul(this.b.T());
        gradB = this.a.T().matmul(gradB);

        this.a.accumulate_grad(gradA);
        this.b.accumulate_grad(gradB);
    }
}