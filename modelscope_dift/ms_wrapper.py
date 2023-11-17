# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS


@MODELS.register_module('modelscope_dift', module_name='my-custom-model')
class MyCustomModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        from modelscope_dift.dift_sd import SDFeaturizer
        self.model = SDFeaturizer(self.model_dir, device='cuda')
        from torchvision.transforms import PILToTensor
        self.transform = PILToTensor()

    def forward(self, image, **forward_params):
        img = LoadImage.convert_to_img(image)
        img_size = forward_params.pop("img_size", [768, 768])
        if not isinstance(img_size, list):
            img_size = [img_size, img_size]
        if img_size[0] > 0:
            img = img.resize(img_size)
        img_tensor = (self.transform(img) / 255.0 - 0.5) * 2
        return self.model.forward(img_tensor, **forward_params)


@PREPROCESSORS.register_module('modelscope_dift', module_name='my-custom-preprocessor')
class MyCustomPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainsforms = self.init_preprocessor(**kwargs)

    def __call__(self, results):
        return self.trainsforms(results)

    def init_preprocessor(self, **kwarg):
        """ Provide default implementation based on preprocess_cfg and user can reimplement it.
            if nothing to do, then return lambda x: x
        """
        return lambda x: x


@PIPELINES.register_module('modelscope_dift', module_name='my-custom-pipeline')
class MyCustomPipeline(Pipeline):

    def __init__(self, model, preprocessor=None, **kwargs):
        """
        use `model` and `preprocessor` to create a custom pipeline for prediction
        Args:
            model: model id on modelscope hub.
            preprocessor: the class of method be init_preprocessor
        """
        super().__init__(model=model, auto_collate=False)
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or Model'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.eval()

        if preprocessor is None:
            preprocessor = MyCustomPreprocessor()
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}

    def _check_input(self, inputs):
        pass

    def _check_output(self, outputs):
        pass

    def forward(self, inputs, **forward_params):
        return super().forward(inputs, **forward_params)

    def postprocess(self, inputs):
        return inputs

# Tips: usr_config_path is the temporary save configuration location， after upload modelscope hub, it is the model_id
# usr_config_path = '/tmp/snapdown/'
# config = Config({
#     "framework": 'pytorch',
#     "task": 'modelscope_dift',
#     "model": {'type': 'my-custom-model'},
#     "pipeline": {"type": "my-custom-pipeline"},
#     "allow_remote": True
# })
# config.dump('/tmp/snapdown/' + 'configuration.json')

# if __name__ == "__main__":
#     from modelscope.models import Model
#     from modelscope.pipelines import pipeline
#     # model = Model.from_pretrained(usr_config_path)
#     input = "./assets/cat.png"
#     inference = pipeline('modelscope_dift', model=usr_config_path)
#     output = inference(input, img_size=0, t=261, up_ft_index=1, prompt='a photo of a cat', ensemble_size=8, seed=0)
#     print(output)
