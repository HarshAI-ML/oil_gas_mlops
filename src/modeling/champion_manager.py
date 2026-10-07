from mlflow.tracking import MlflowClient


class ChampionChallengerManager:

    def __init__(self, model_name, champion_alias="champion", challenger_alias="challenger"):
        self.model_name = model_name
        self.champion_alias = champion_alias
        self.challenger_alias = challenger_alias
        self.client = MlflowClient()

    def _get_rmse_by_alias(self, alias):
        # Returns (version, validation_rmse) for whichever version holds this alias
        # Returns (None, None) if the alias does not exist yet
        try:
            version = self.client.get_model_version_by_alias(self.model_name, alias)
        except Exception:
            return None, None

        run = self.client.get_run(version.run_id)
        rmse = run.data.metrics.get("validation_rmse")
        return version.version, rmse

    def promote_challenger_if_better(self, new_version, new_rmse):
        # Step 1: the newly registered model always becomes the challenger
        self.client.set_registered_model_alias(self.model_name, self.challenger_alias, new_version)
        print("Version " + str(new_version) + " set as '" + self.challenger_alias + "'")

        # Step 2: does a champion already exist?
        champion_version, champion_rmse = self._get_rmse_by_alias(self.champion_alias)

        if champion_version is None:
            self.client.set_registered_model_alias(self.model_name, self.champion_alias, new_version)
            print("No existing champion. Version " + str(new_version) + " promoted to '" + self.champion_alias + "'")
            return

        # Step 3: compare the challenger's RMSE against the champion's RMSE
        print("Champion (v" + str(champion_version) + ") validation RMSE: " + str(champion_rmse))
        print("Challenger (v" + str(new_version) + ") validation RMSE: " + str(new_rmse))

        if new_rmse < champion_rmse:
            self.client.set_registered_model_alias(self.model_name, self.champion_alias, new_version)
            print("Version " + str(new_version) + " promoted to '" + self.champion_alias + "'")
        else:
            print("Version " + str(new_version) + " remains '" + self.challenger_alias +
                  "', current champion (v" + str(champion_version) + ") is not beaten")
