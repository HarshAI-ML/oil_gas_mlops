from modeling.registrar import ModelRegistrar


def test_returns_none():

    registrar_model= ModelRegistrar(model_name="test_model",max_rmse=1000,min_r2=0.85)

    registered_model=registrar_model.register_if_good(model_uri="runs:/abc/model",rmse=1500,r2=0.90)

    assert registered_model is None

