import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator_testing(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.testing_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def _valid_generator_training(self):
        datagenerator_kwargs = dict(
            rescale=1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode='categorical',
             classes={'Normal': 0, 
                     'Viral Pneumonia': 1,
                     'Covid': 2},
            **dataflow_kwargs

        )
       




    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)

        self._valid_generator_testing()
        testing_score = self.model.evaluate(self.valid_generator)
        self.save_score(testing_score, "testing_scores.json")

        # Evaluate on training data
        self._valid_generator_training()
        training_score = self.model.evaluate(self.valid_generator)
        self.save_score(training_score, "training_scores.json")

        self.log_into_mlflow(testing_score, training_score)


  

    def save_score(self, score, filename):
        scores = {"loss": score[0], "accuracy": score[1]}
        save_json(path=Path(filename), data=scores)

    
    def log_into_mlflow(self,training_score,testing_score):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
            {"testing_loss": testing_score[0], "testing_accuracy": testing_score[1],
             "training_loss": training_score[0], "training_accuracy": training_score[1]}
            )
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
        else:
            mlflow.keras.log_model(self.model, "model")