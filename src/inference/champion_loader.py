import mlflow.sklearn
from mlflow.tracking import MlflowClient


class ChampionLoader:

    def __init__(self, model_name, alias="champion"):
        self.model_name = model_name
        self.alias = alias
        self.client = MlflowClient()

    def get_version(self):
        champion = self.client.get_model_version_by_alias(self.model_name, self.alias)
        return champion.version

    def load_model(self):
        model_uri = "models:/" + self.model_name + "@" + self.alias
        return mlflow.sklearn.load_model(model_uri)