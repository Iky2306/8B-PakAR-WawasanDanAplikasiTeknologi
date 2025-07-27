# Framework Implementasi Bertanggung Jawab

# 1. Explainable AI (XAI) untuk Transparansi
def generate_explanation(model, image, layer_name='conv5'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
    return np.maximum(cam, 0) / np.max(cam)

# 2. Sistem Pemantauan Bias Real-time
from aif360.metrics import BinaryLabelDatasetMetric

def calculate_disparity(model, test_data, protected_attribute):
    predictions = model.predict(test_data.features)
    dataset = test_data.copy(predicted_labels=predictions)
    
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{protected_attribute: 1}],
        unprivileged_groups=[{protected_attribute: 0}])
    
    return metric.mean_difference()
