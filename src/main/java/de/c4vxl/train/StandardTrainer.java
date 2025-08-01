package de.c4vxl.train;

import de.c4vxl.core.nn.loss.type.LossFunction;
import de.c4vxl.core.nn.module.Module;
import de.c4vxl.core.optim.type.AbstractOptimizer;
import de.c4vxl.core.optim.type.Optimizer;
import de.c4vxl.core.tensor.Tensor;
import de.c4vxl.core.type.DType;
import de.c4vxl.models.type.Model;
import de.c4vxl.train.type.Trainer;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * This trainer allows for training on most preprocessed datasets with a train and an optional test split.
 */
public class StandardTrainer<T> extends Trainer {
    private final HashMap<Tensor<T>, Tensor<T>> trainSplit, testSplit;
    private final Model<T> model;
    private final int num_epochs, logging_rate, testing_rate;
    private final Optimizer optimizer;
    private final LossFunction criterion;
    private final BiConsumer<Integer, Double> trainLogger, testLogger;
    private final Double clip_grads_min, clip_grads_max;

    /**
     * Create a new trainer instance without a testing dataset split
     * @param model The model to train
     * @param optimizer The optimizer connected to the models parameters
     * @param criterion The loss function to evaluate model performance
     * @param trainSplit The train split of the dataset
     * @param num_epochs The amount of epochs of training
     * @param logging_rate The frequency of calling the {@code trainLogger}
     * @param trainLogger A function for handling the loss in an epoch that is a multiple of {@code logging_rate}
     * @param clip_grads_min The minimum value gradients can be. (Set to null to ignore)
     * @param clip_grads_max The maximum value gradients can be. (Set to null to ignore)
     */
    public StandardTrainer(
            Model<T> model,
            Optimizer optimizer,
            LossFunction criterion,
            HashMap<Tensor<T>, Tensor<T>> trainSplit,
            int num_epochs,
            int logging_rate,
            BiConsumer<Integer, Double> trainLogger,
            Double clip_grads_min,
            Double clip_grads_max
    ) {
        this(model, optimizer, criterion, trainSplit, null, num_epochs, logging_rate, -1, trainLogger, null, clip_grads_min, clip_grads_max);
    }

    /**
     * Create a new trainer instance
     * @param model The model to train
     * @param optimizer The optimizer connected to the models parameters
     * @param criterion The loss function to evaluate model performance
     * @param trainSplit The train split of the dataset
     * @param testSplit The test split of the dataset
     * @param num_epochs The amount of epochs of training
     * @param logging_rate The frequency of calling the {@code trainLogger}
     * @param testing_rate The frequency of calling the {@code testLogger}
     * @param trainLogger A function for handling the loss in an epoch that is a multiple of {@code logging_rate}
     * @param testLogger A function for handling the loss in an epoch that is a multiple of {@code testing_rate}
     * @param clip_grads_min The minimum value gradients can be. (Set to null to ignore)
     * @param clip_grads_max The maximum value gradients can be. (Set to null to ignore)
     */
    public StandardTrainer(
            Model<T> model,
            Optimizer optimizer,
            LossFunction criterion,
            HashMap<Tensor<T>, Tensor<T>> trainSplit,
            HashMap<Tensor<T>, Tensor<T>> testSplit,
            int num_epochs,
            int logging_rate,
            int testing_rate,
            BiConsumer<Integer, Double> trainLogger,
            BiConsumer<Integer, Double> testLogger,
            Double clip_grads_min,
            Double clip_grads_max
    ) {
        this.model = model;
        this.optimizer = optimizer;
        this.criterion = criterion;
        this.trainSplit = trainSplit;
        this.testSplit = testSplit;
        this.num_epochs = num_epochs;
        this.logging_rate = logging_rate;
        this.testing_rate = testing_rate;
        this.trainLogger = trainLogger;
        this.testLogger = testLogger;
        this.clip_grads_min = clip_grads_min;
        this.clip_grads_max = clip_grads_max;

        if (clip_grads_min != null && clip_grads_max != null && !(optimizer instanceof AbstractOptimizer))
            System.err.println("[WARNING]: Gradients won't be clipped due to missing support by optimizer. Please use an 'AbstractOptimizer'.");
    }

    @Override
    public void train() {
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double train_loss = iteration(trainSplit, true);
            if (trainLogger != null && epoch % logging_rate == 0)
                trainLogger.accept(epoch, train_loss);

            if (testSplit != null && testing_rate != -1 && epoch % testing_rate == 0) {
                double test_loss = iteration(testSplit, false);
                if (testLogger != null)
                    testLogger.accept(epoch, test_loss);
            }
        }
    }

    private double iteration(HashMap<Tensor<T>, Tensor<T>> dataset, boolean isTrain) {
        if (isTrain)
            model.train();
        else
            model.eval();

        double total_loss = 0.;
        for (Map.Entry<Tensor<T>, Tensor<T>> batch : dataset.entrySet()) {
            // Make prediction
            Tensor<T> inputs = batch.getKey().detach(), targets = batch.getValue().detach();
            Tensor<T> predictions = model.forward(inputs);

            // Compute loss
            Tensor<T> loss = criterion.forward(predictions, targets);

            // Optimize weights
            if (isTrain) {
                optimizer.zeroGrad();
                loss.backward();

                if (clip_grads_min != null && clip_grads_max != null && optimizer instanceof AbstractOptimizer ao)
                    ao.clip_gradients(clip_grads_min, clip_grads_max);

                optimizer.step();
            }

            total_loss += DType.DOUBLE.parse(loss.item());
        }

        return total_loss;
    }
}