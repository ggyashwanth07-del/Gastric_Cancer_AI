
import tensorflow as tf

model = tf.keras.models.load_model(
    "model/efficientnet_b3.keras",
    compile=False
)

model(tf.zeros((1, 300, 300, 3)), training=False)

def show_layers(current_model, prefix=""):
    for layer in current_model.layers:
        print(f"{prefix}{layer.name} : {type(layer).__name__}")

        if isinstance(layer, tf.keras.Model):
            show_layers(layer, prefix + layer.name + "/")

print("\n===== INTERNAL LAYER NAMES =====")
show_layers(model)