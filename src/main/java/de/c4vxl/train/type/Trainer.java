package de.c4vxl.train.type;

import de.c4vxl.core.nn.loss.type.LossFunction;

public abstract class Trainer {
    public abstract void train();

    /**
     * Returns a loss function from its name
     * @param name The name of the loss function
     */
    @SuppressWarnings("unchecked")
    public static LossFunction getLossFunction(String name) {
        try {
            Class<LossFunction> clazz = (Class<LossFunction>) Class.forName("de.c4vxl.core.nn.loss." + name);
            return clazz.getConstructor().newInstance();
        } catch (Exception e) {
            if (!name.toLowerCase().endsWith("loss"))
                return getLossFunction(name + "Loss");

            return null;
        }
    }
}