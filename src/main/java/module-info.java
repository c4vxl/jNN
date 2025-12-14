module jNN.main {
    // Require external modules
    requires com.google.gson;
    requires nd4j.api;

    // main
    exports de.c4vxl;

    // core.nn
    exports de.c4vxl.core.nn;
    exports de.c4vxl.core.nn.activation;
    exports de.c4vxl.core.nn.activation.type;
    exports de.c4vxl.core.nn.loss;
    exports de.c4vxl.core.nn.loss.type;
    exports de.c4vxl.core.nn.module;

    // core.optim
    exports de.c4vxl.core.optim;
    exports de.c4vxl.core.optim.type;

    // core.tensor
    exports de.c4vxl.core.tensor;
    exports de.c4vxl.core.tensor.grad;
    exports de.c4vxl.core.tensor.operation;
    exports de.c4vxl.core.tensor.operation.type;

    // core.type
    exports de.c4vxl.core.type;

    // core.utils
    exports de.c4vxl.core.utils;

    // models
    exports de.c4vxl.models;
    exports de.c4vxl.models.type;

    // pipeline
    exports de.c4vxl.pipeline;
    exports de.c4vxl.pipeline.type;

    // tokenizers
    exports de.c4vxl.tokenizers;
    exports de.c4vxl.tokenizers.type;

    // train
    exports de.c4vxl.train;
    exports de.c4vxl.train.type;
}