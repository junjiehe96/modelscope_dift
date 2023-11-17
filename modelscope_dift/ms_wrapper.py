# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS


@MODELS.register_module('feature_extraction', module_name='my-custom-model')
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
        if not isinstance(img_size, list) and not isinstance(img_size, tuple):
            img_size = [img_size, img_size]
        if img_size[0] > 0:
            img = img.resize(img_size)
        img_tensor = (self.transform(img) / 255.0 - 0.5) * 2
        return self.model.forward(img_tensor, **forward_params)


@PREPROCESSORS.register_module('feature_extraction', module_name='my-custom-preprocessor')
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


@PIPELINES.register_module('feature_extraction', module_name='my-custom-pipeline')
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
        out = {
            OutputKeys.OUTPUT: super().forward(inputs, **forward_params)
        }
        return out

    def postprocess(self, inputs):
        return inputs


if __name__ == "__main__":
    from modelscope.pipelines import pipeline

    model = 'damo/cv_stable-diffusion-v2_image-feature'
    pipe = pipeline('feature_extraction', model=model, device='gpu', auto_collate=False, model_revision='v1.0.1')
    out1 = pipe('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_recognition_1.png', img_size=0,
                t=261, up_ft_index=2, prompt='a photo of a girl', ensemble_size=4, seed=0)[OutputKeys.OUTPUT]  # 1*C*H*W
    print(out1)
