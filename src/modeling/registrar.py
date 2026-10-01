# src/modeling/registrar.py
import mlflow


class ModelRegistrar:

    def __init__(self, model_name, max_rmse, min_r2):
        self.model_name = model_name
        self.max_rmse = max_rmse
        self.min_r2 = min_r2

    def register_if_good(self, model_uri, rmse, r2):
        # Returns the new version number, or None if thresholds were not met
        rmse_ok = rmse < self.max_rmse
        r2_ok = r2 > self.min_r2

        if rmse_ok and r2_ok:
            registered = mlflow.register_model(model_uri=model_uri, name=self.model_name)
            print("Registered as version " + str(registered.version))
            return registered.version

        print("Not registered. Thresholds not met:")
        print("  RMSE: " + str(rmse) + " (needs < " + str(self.max_rmse) + ") -> " + ("OK" if rmse_ok else "FAIL"))
        print("  R2: " + str(r2) + " (needs > " + str(self.min_r2) + ") -> " + ("OK" if r2_ok else "FAIL"))
        return None
