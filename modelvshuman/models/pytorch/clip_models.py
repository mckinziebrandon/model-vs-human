#!/usr/bin/env python3
import torch

from ..registry import register_model
from ..wrappers.pytorch import (
    OpenAIClipPytorchModelWrapper,
    SIMLClipPytorchModelWrapper,
    CybertronClipPytorchModelWrapper,
)

__all__ = [
    'clip_base',
    'clip_large',
    'clip_large_336',
    'clipRN50',
    'cybertron_clip_base',
    'cybertron_clip_large',
    'cybertron_clip_base_simple_prompts',
    'cybertron_clip_large_simple_prompts',
    'cybertron_clip_base_shape_prompts',
    'cybertron_clip_large_shape_prompts',
    'cybertron_clip_base_texture_prompts',
    'cybertron_clip_large_texture_prompts',
]


def make_cybertron_base_prompt_model_fn(prompts):
    def model_fn(model_name, *args):
        checkpoint_path = 's3://cybertron/artifacts/checkpoints/yinfei_yang/' \
                          'CLIP_ALIGN_config_all_data_0303/permanent_checkpoint_000600000.pt'
        return CybertronClipPytorchModelWrapper(
            checkpoint_path, model_name, *args, prompts=prompts)
    return model_fn


def make_cybertron_large_prompt_model_fn(prompts):
    def model_fn(model_name, *args):
        checkpoint_path = 's3://cybertron/artifacts/checkpoints/yinfei_yang/' \
                          'CLIP_large_lr5e-4_eps1e-6_0508/permanent_checkpoint_000234000.pt'
        return CybertronClipPytorchModelWrapper(
            checkpoint_path, model_name, *args, prompts=prompts)
    return model_fn


def make_openai_prompt_model_fn(clip_model_name, prompts):
    def model_fn(model_name, *args):
        import clip
        model, _ = clip.load(clip_model_name)
        return OpenAIClipPytorchModelWrapper(
            model, model_name, *args, prompts=prompts)
    return model_fn



@register_model("pytorch")
def clip_base(model_name, *args):
    import clip
    model, _ = clip.load("ViT-B/16")
    return OpenAIClipPytorchModelWrapper(model, model_name, *args)


@register_model("pytorch")
def clip_large(model_name, *args):
    import clip
    model, _ = clip.load("ViT-L/14")
    return OpenAIClipPytorchModelWrapper(model, model_name, *args)


@register_model("pytorch")
def clip_large_336(model_name, *args):
    import clip
    model, _ = clip.load("ViT-L/14@336px")
    return OpenAIClipPytorchModelWrapper(model, model_name, *args)


@register_model("pytorch")
def clipRN50(model_name, *args):
    import clip
    model, _ = clip.load("RN50")
    return OpenAIClipPytorchModelWrapper(model, model_name, *args)


@register_model("pytorch")
def cybertron_clip_base(model_name, *args):
    checkpoint_path = 's3://cybertron/artifacts/checkpoints/yinfei_yang/' \
                      'CLIP_ALIGN_config_all_data_0303/permanent_checkpoint_000600000.pt'
    return CybertronClipPytorchModelWrapper(checkpoint_path, model_name, *args)


@register_model("pytorch")
def cybertron_clip_large(model_name, *args):
    checkpoint_path = 's3://cybertron/artifacts/checkpoints/yinfei_yang/' \
                      'CLIP_large_lr5e-4_eps1e-6_0508/permanent_checkpoint_000234000.pt'
    return CybertronClipPytorchModelWrapper(checkpoint_path, model_name, *args)


SIMPLE_PROMPTS = [
   'a photo of a {}.',
]


@register_model("pytorch")
def clip_base_simple_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-B/16', SIMPLE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_large_simple_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14', SIMPLE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_large336px_simple_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14@336px', SIMPLE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def cybertron_clip_base_simple_prompts(model_name, *args):
    return make_cybertron_base_prompt_model_fn(SIMPLE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def cybertron_clip_large_simple_prompts(model_name, *args):
    return make_cybertron_large_prompt_model_fn(SIMPLE_PROMPTS)(model_name, *args)


SHAPE_PROMPTS = [
    'the shape of a {}.',
    'a photo in the shape of a {}.',
    'a sketch in the shape of a {}.',
    'a drawing in the shape of a {}.',
    'a painting in the shape of a {}.',
    'a sculpture in the shape of a {}.',
]


@register_model("pytorch")
def clip_base_shape_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-B/16', SHAPE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_large_shape_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14', SHAPE_PROMPTS)(model_name, *args)

@register_model("pytorch")
def clip_large336px_shape_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14@336px', SHAPE_PROMPTS)(model_name, *args)

@register_model("pytorch")
def cybertron_clip_base_shape_prompts(model_name, *args):
    return make_cybertron_base_prompt_model_fn(SHAPE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def cybertron_clip_large_shape_prompts(model_name, *args):
    return make_cybertron_large_prompt_model_fn(SHAPE_PROMPTS)(model_name, *args)

TEXTURE_PROMPTS = [
    p.replace('shape', 'texture') for p in SHAPE_PROMPTS
]

@register_model("pytorch")
def cybertron_clip_base_texture_prompts(model_name, *args):
    return make_cybertron_base_prompt_model_fn(TEXTURE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def cybertron_clip_large_texture_prompts(model_name, *args):
    return make_cybertron_large_prompt_model_fn(TEXTURE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_base_texture_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-B/16', TEXTURE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_large_texture_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14', TEXTURE_PROMPTS)(model_name, *args)


@register_model("pytorch")
def clip_large336px_texture_prompts(model_name, *args):
    return make_openai_prompt_model_fn('ViT-L/14@336px', TEXTURE_PROMPTS)(model_name, *args)
