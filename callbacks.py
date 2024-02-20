import keras
import keras_cv
# pylint: disable=not-callable


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """Evaluates BoxCOCO Metrics for given dataset"""

    def __init__(self, data, save_path):
        """Initializes EvaluateCOCOMetricsCallback object and creates BoxCOCOMetrics object

        Args:
            data (_type_): tf typed data to evaluate
            save_path (str): best model save path
        """

        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        """Calculate metrics on each epoch end

        Args:
            epoch (int): number of the past epoch
            logs (str): logs of training

        Returns:
            _type_: _description_
        """
        
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs
