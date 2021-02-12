from joblib import dump, load
from .DNN import DNNVanilla

def save_numerai_run(conf, trained_output):
    dump(trained_output, conf.model_save_file)

def load_numerai_run(model_loc, trained_output):
    load(model_loc)

class NumeraiTrainedOutput:

    def __init__(self, features, trained_models, ensemble_weights, run_param, test_param):

        self.features = features
        self.trained_models = trained_models.copy()
        self.ensemble_weights = ensemble_weights
        self.run_param = run_param
        self.test_param = test_param

        self.export_keras_models()

    def export_keras_models(self):
        for i in self.trained_models.keys():
            if i.startswith('DNN'):
                self.trained_models[i] = self.trained_models[i].to_json()

    def get_output(self):
        return {'features': self.features,
                'trained_models': self.trained_models,
                'ensemble_weights': self.ensemble_weights,
                'run_param': self.run_param,
                'test_param': self.test_param}

    def convert_keras_from_json(self):

        for i in self.trained_models.keys():
            if i.startswith('DNN'):
                dnn_mod = DNNVanilla()
                dnn_mod.from_json(self.trained_models[i])
                self.trained_models[i] = dnn_mod
