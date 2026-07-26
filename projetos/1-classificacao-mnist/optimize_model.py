import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    # Carrega o modelo treinado
    model = tf.keras.models.load_model(model_path)

    # Converte para TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aplica Dynamic Range Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Salva o resultado
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(model_path) / 1024
    optimized_size = os.path.getsize(tflite_path) / 1024

    print(f"Modelo original (model.h5): {original_size:.2f} KB")
    print(f"Modelo otimizado (model.tflite): {optimized_size:.2f} KB")
    print(f"Reducao de tamanho: {(1 - optimized_size/original_size) * 100:.1f}%")
    print("\nConversao concluida! Modelo salvo como model.tflite")


if __name__ == "__main__":
    main()