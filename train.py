import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report, f1_score, confusion_matrix 

def compile_and_train(model, dataset, epochs, buffer_size, batch_size):
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
    
    history = model.fit(dataset, epochs=epochs)
    return history

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset, num=1):
    all_true_masks = []
    all_pred_masks = []
    
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        pred_mask_processed = create_mask(pred_mask)

        all_true_masks.append(mask[0].numpy().flatten())
        all_pred_masks.append(pred_mask_processed.numpy().flatten())

        display([image[0], mask[0], pred_mask_processed])

    true_masks_flat = tf.concat(all_true_masks, axis=0)
    pred_masks_flat = tf.concat(all_pred_masks, axis=0)
    
    f1 = f1_score(true_masks_flat, pred_masks_flat, average='weighted')
    cm = confusion_matrix(true_masks_flat, pred_masks_flat)

    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)