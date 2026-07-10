import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import SwinConfig, TFSwinForImageClassification

import wandb
from wandb.keras import WandbMetricsLogger

def get_args():
    parser = argparse.ArgumentParser(description="Classification Benchmark API")
    
    # Core CLI Parameters
    parser.add_argument(
        "--model", 
        type=str, 
        default="resnet", 
        choices=["resnet", "densenet", "swin"],
        help="Target model architecture for training"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cifar10", 
        choices=["cifar10"],
        help="Target dataset for evaluation"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of full training epochs"
    )
    
    # Optional Hyperparameters (Defaults Maintained)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    
    return parser.parse_args()

def preprocess_data(images, labels):
    # Upscale images to 224x224 and normalize values to [0, 1]
    images = tf.image.resize(images, (224, 224))
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

def build_model(model_name):
    inputs = layers.Input(shape=(224, 224, 3))
    if model_name == "resnet":
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=inputs)
        x = layers.GlobalAveragePooling2D()(base.output)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
    elif model_name == "densenet":
        base = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
        x = layers.GlobalAveragePooling2D()(base.output)
        outputs = layers.Dense(10, activation="softmax")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
    elif model_name == "swin":
        # Initialize an un-pretrained Swin architecture layout locally
        config = SwinConfig(image_size=224, num_labels=10, num_channels=3)
        transformer_model = TFSwinForImageClassification(config)
        
        # Wrap Transformer outputs to comply with standard Keras training pipeline
        outputs = transformer_model(inputs).logits
        outputs = layers.Activation("softmax")(outputs)
        model = models.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    args = get_args()

    # Initialize the Weights & Biases experiment track tracking context
    wandb.init(
        project="cifar10-benchmark",
        name=f"tensorflow-{args.model}",
        config=vars(args) # Logs hyperparameters dynamically
    )

    print(f"Target Configuration | Model: {args.model} | Device Availability: {tf.config.list_physical_devices('GPU')}")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Formulate performance-optimized tf.data Pipelines
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=50000)
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(args.model)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_dataset, epochs=args.epochs, validation_data=test_dataset, callbacks=[WandbMetricsLogger(log_freq="epoch")])

    # Record final validation accuracy and close active runner thread safely
    wandb.finish()  

if __name__ == "__main__":
    main()