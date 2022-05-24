import torch.nn as nn
# Disable warnings to prevent cybertron from logging a ton of them.
import logging
logging.disable(logging.WARNING)
logger = logging.getLogger('modelvshuman')

from typing import List
import math
import PIL
import clip
import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from tqdm import tqdm

from .base import AbstractModel
from ..pytorch.clip.imagenet_classes import imagenet_classes
from ..pytorch.clip.imagenet_templates import imagenet_templates


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""

    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

    return images


class PytorchModel(AbstractModel):

    def __init__(self, model, model_name, *args):
        self.model = model
        self.model_name = model_name
        self.args = args
        self.model.to(device())

    def to_numpy(self, x):
        if x.is_cuda:
            return x.detach().cpu().numpy()
        else:
            return x.numpy()

    def softmax(self, logits):
        assert type(logits) is np.ndarray

        softmax_op = torch.nn.Softmax(dim=1)
        softmax_output = softmax_op(torch.Tensor(logits))
        return self.to_numpy(softmax_output)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        self.model.eval()
        logits = self.model(images)
        return self.to_numpy(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class PyContrastPytorchModel(PytorchModel):
    """
    This class inherits PytorchModel class to adapt model validation for Pycontrast pre-trained models from
    https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, model, classifier, model_name, *args):
        super().__init__(model, model_name, args)
        self.classifier = classifier
        self.classifier.to(device())

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        self.classifier.eval()
        feat = self.model(images, mode=2)
        output = self.classifier(feat)
        return self.to_numpy(output)


class ViTPytorchModel(PytorchModel):

    def __init__(self, model, model_name, img_size=(384, 384), *args):
        self.img_size = img_size
        super().__init__(model, model_name, args)

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)

    def preprocess(self):
        # custom preprocessing from:
        # https://github.com/lukemelas/PyTorch-Pretrained-ViT

        return Compose([
            Resize(self.img_size),
            ToTensor(),
            Normalize(0.5, 0.5),
        ])



class EfficientNetPytorchModel(PytorchModel):

    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name, *args)

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        img_size = 475
        crop_pct = 0.936
        scale_size = int(math.floor(img_size / crop_pct)) 
        return Compose([
            Resize(scale_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())

        logits = self.model(images)
        return self.to_numpy(logits)


class SwagPytorchModel(PytorchModel):

    def __init__(self, model, model_name, input_size, *args):
        super().__init__(model, model_name, *args)
        self.input_size = input_size

    def preprocess(self):
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        return Compose([
            Resize(self.input_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(self.input_size),
            ToTensor(),
            normalize,
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor
        self.model.eval()
        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0)).to(device())
        logits = self.model(images)
        return self.to_numpy(logits)


class ConvertToRGB:
    def __call__(self, x):
        return x.convert("RGB")


CYBERTRON_CKPTS = {
    'ViT-B/16': 's3://cybertron/artifacts/checkpoints/yinfei_yang'
                '/CLIP_ALIGN_config_all_data_0303/permanent_checkpoint_000600000.pt',
    'ViT-L/14': 's3://cybertron/artifacts/checkpoints/yinfei_yang/CLIP_large_lr5e'
                '-4_eps1e-6_0508/permanent_checkpoint_000234000.pt',
}


class BaseCLIPWrapper(PytorchModel):
    def __init__(
            self,
            model: nn.Module,
            model_name: str,
            prompts: List[str] = imagenet_templates,
            *args
    ):
        super().__init__(model, model_name, *args)
        self.prompts = prompts
        self.zeroshot_weights = self._get_zeroshot_weights(imagenet_classes)

    def encode_text_batch(self, text_batch: List[str]) -> torch.Tensor:
        """Subclasses must implement."""
        raise NotImplementedError()

    @torch.no_grad()
    def _get_zeroshot_weights(self, class_names):
        zeroshot_weights = []
        for class_name in tqdm(class_names, desc="class_names"):
            texts = [
                prompt.format(class_name) for prompt in self.prompts]
            class_embeddings = self.encode_text_batch(texts)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device())
        return zeroshot_weights


class OpenAIClipPytorchModelWrapper(BaseCLIPWrapper):

    def __init__(self, model, model_name, prompts, *args):
        super().__init__(model, model_name, prompts, *args)

    @torch.no_grad()
    def encode_text_batch(self, text_batch: List[str]) -> torch.Tensor:
        texts = clip.tokenize(text_batch).to(device())  # tokenize
        class_embeddings = self.model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        return class_embeddings

    def preprocess(self, n_px=224):
        return Compose([
            Resize(n_px, interpolation=PIL.Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        transform = self.preprocess(self.model.visual.input_resolution)
        images = [transform(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0))

        self.model.eval()

        image_features = self.model.encode_image(images.to(device()))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)


class SIMLClipPytorchModelWrapper(BaseCLIPWrapper):

    def __init__(self, wrapper, model_name, prompts, *args):
        super().__init__(wrapper.model, model_name, prompts, *args)
        self.wrapper = wrapper

    @torch.no_grad()
    def encode_text_batch(self, text_batch: List[str]) -> torch.Tensor:
        return self.wrapper.encode_text_batch_from_str(text_batch)

    def preprocess(self):
        return Compose([
            Resize((self.wrapper.image_size, self.wrapper.image_size),
                   interpolation=PIL.Image.BILINEAR),
            ConvertToRGB(),
            ToTensor(),
        ])

    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        images = [self.preprocess()(ToPILImage()(image)) for image in images]
        images = torch.Tensor(np.stack(images, axis=0))
        self.model.eval()

        image_features = self.wrapper.encode_image_batch(images.to(device()))
        logits = 100. * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)


class CybertronClipPytorchModelWrapper(BaseCLIPWrapper):

    def __init__(
            self,
            checkpoint_path: str,
            model_name: str,
            prompts: List[str] = imagenet_templates,
            *args
    ):
        from cybertron.images.clip.datasets import CLIPDataCollator
        from cybertron.data.checkpoint import call_with_checkpoint_config, \
            load_checkpoint
        # from cybertron import configure_logger
        # from cybertron import io as cio
        # logger = logging.getLogger('CLIPViewer')
        # configure_logger()
        model = load_checkpoint(
            path=checkpoint_path,
            model=None,
            device='cuda').eval()
        self.data_collator = call_with_checkpoint_config(
            CLIPDataCollator,
            checkpoint_path,
            suppress_gin_bindings=[
                'default_image_transform_pipeline.eval_mode = False'],
            add_gin_bindings=['default_image_transform_pipeline.eval_mode = True'])
        super().__init__(model, model_name, prompts, *args)

    @torch.no_grad()
    def encode_text_batch(self, text_batch: List[str]) -> torch.Tensor:
        collator_inputs = [
            {'text': t.encode('utf-8')} for t in text_batch]
        text_inputs = {
            k: v.to(device()) for k, v in
            self.data_collator(collator_inputs).items()}
        return self.model.encode_text(**text_inputs)

    @torch.no_grad()
    def forward_batch(self, images):
        assert type(images) is torch.Tensor

        images = undo_default_preprocessing(images)
        image_tensors = torch.cat([
            self.data_collator._img_transform(ToPILImage()(im))[None, ...] for im in images
        ], dim=0)
        image_masks = torch.cat([
            torch.ones(1, self.data_collator._grid_size ** 2, dtype=torch.int32) for _ in
            images
        ], dim=0)
        self.model.eval()

        image_features = self.model.encode_image(
            images=image_tensors.to(device()),
            image_masks=image_masks.to(device()))
        logits = 100. * image_features @ self.zeroshot_weights
        return self.to_numpy(logits)
