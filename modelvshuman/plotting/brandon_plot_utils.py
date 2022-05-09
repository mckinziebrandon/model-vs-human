
from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker

blue = rgb(0, 102, 102)
purple = rgb(76, 0, 153)
grey2 = rgb(110, 110, 110)
bitm_col = rgb(153, 142, 195)

SHAPES = {
    'circle': 'o',
    'square': 's',
    'star': '*',
    'triangle_down': 'v',
    'triangle_up': '^',
    'diamond': 'D',
}
COLORS = {
    'vit': rgb(144, 159, 110),
}
DECISION_MAKERS = {
    # CLIP
    'siml_clip': lambda df: DecisionMaker(
        name_pattern="siml_clip",
        color=rgb(65, 90, 140), marker="o", df=df,
        plotting_name="SIML CLIP"),
    'siml_clip_pruned': lambda df: DecisionMaker(
        name_pattern="siml_clip_pruned",
        color=rgb(65, 90, 140), marker=SHAPES['diamond'], df=df,
        plotting_name="SIML CLIP Pruned"),
    'clip_base': lambda df: DecisionMaker(
        name_pattern="clip_base",
        color=grey2, marker=SHAPES['circle'], df=df,
        plotting_name="CLIP ViT-B/16"),
    'clip_large': lambda df: DecisionMaker(
        name_pattern="clip_large",
        color=grey2, marker=SHAPES['triangle_up'], df=df,
        plotting_name="CLIP ViT-L/14"),
    'clip_large_336': lambda df: DecisionMaker(
        name_pattern="clip_large_336",
        color=gold, marker=SHAPES['star'], df=df,
        plotting_name="CLIP ViT-L/14@336px"),
    'clipRN50': lambda df: DecisionMaker(
        name_pattern="clipRN50",
        color=grey1, marker=".", df=df,
        plotting_name="CLIP RN50"),
    # ViT
    'vit_small_patch16_224': lambda df: DecisionMaker(
        name_pattern="vit_small_patch16_224",
        color=COLORS['vit'], marker="v", df=df,
        plotting_name="ViT-S"),
    'vit_base_patch16_224': lambda df: DecisionMaker(
        name_pattern="vit_base_patch16_224",
        color=COLORS['vit'], marker="v", df=df,
        plotting_name="ViT-B"),
    'vit_large_patch16_224': lambda df: DecisionMaker(
        name_pattern="vit_large_patch16_224",
        color=COLORS['vit'], marker="v", df=df,
        plotting_name="ViT-L"),
    # SWSL
    'ResNeXt101_32x16d_swsl': lambda df: DecisionMaker(
        name_pattern="ResNeXt101_32x16d_swsl",
        color=purple, marker=SHAPES['diamond'], df=df,
        plotting_name="ResNeXt101"),
    'resnet50_swsl': lambda df: DecisionMaker(
        name_pattern="resnet50_swsl",
        color=purple1, marker="o", df=df,
        plotting_name="SWSL: ResNet-50 (940M)"),
    # SimCLR
    'simclr_resnet50x1': lambda df: DecisionMaker(
        name_pattern="simclr_resnet50x1",
        color=orange2, marker="o", df=df,
        plotting_name="SimCLR: ResNet-50x1"),
    'simclr_resnet50x2': lambda df: DecisionMaker(
        name_pattern="simclr_resnet50x2",
        color=orange2, marker="o", df=df,
        plotting_name="SimCLR: ResNet-50x2"),
    'simclr_resnet50x4': lambda df: DecisionMaker(
        name_pattern="simclr_resnet50x4",
        color=orange2, marker="o", df=df,
        plotting_name="SimCLR: ResNet-50x4"),
    # Others
    'efficientnet_l2_noisy_student_475': lambda df: DecisionMaker(
        name_pattern="efficientnet_l2_noisy_student_475",
        color=metallic, marker="o", df=df,
        plotting_name="Noisy Student: ENetL2 (300M)"),
    'BiTM_resnetv2_50x1': lambda df: DecisionMaker(
        name_pattern="BiTM_resnetv2_50x1",
        color=bitm_col, marker="o", df=df,
        plotting_name="BiT-M: ResNet-50x1 (14M)"),
}


def make_brandon_plotting_definition(model_names, extra_decision_makers=None):
    def brandon_plotting_definition(df):
        dms = [
            DECISION_MAKERS[name](df)
            for name in model_names
        ]
        if extra_decision_makers is not None:
            dms.extend(extra_decision_makers)
        dms.append(DecisionMaker(
            name_pattern="subject-*",
            color=rgb(165, 30, 55), marker="D", df=df,
            plotting_name="humans"))
        return dms
    return brandon_plotting_definition
