# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import sys
import cv2

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
)

from fairseq.data.image_dataset import (
    ImageDataset,
    ImagePairDataset,
    ImageGenerator,
)

from . import FairseqTask, register_task
import torchvision.transforms as transforms

from imgaug import augmenters as iaa


""" class ImageAug(object):

    def __init__(self):

        def sometimes(aug):
            return iaa.Sometimes(.90, aug)
        seq = iaa.Sequential(
            [
                sometimes(
                    iaa.CropAndPad(
                        percent=(-0.03, 0.03),
                        pad_mode=["constant", "edge"],
                        pad_cval=(0, 255)
                    )
                ),
                sometimes(
                    iaa.Affine(
                        rotate=(-4, 4),  # rotate by -45 to +45 degrees
                        shear=(-3, 3),  # shear by -16 to +16 degrees
                    )
                ),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.75)),
                    iaa.Dropout((0.01, 0.03), per_channel=0.5)
                ])
            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug """


class ImageAug(object):
    """
    Adds Gaussian noise and other distortions to text.
    """

    def __init__(self):
        def sometimes(aug): return iaa.Sometimes(.50, aug)
        seq = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0.25, 1.0)),  # blur images with a sigma
                    # randomly remove up to n% of the pixels
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                    iaa.CropAndPad(
                        percent=(-0.05, 0.05),
                        pad_mode=["constant"],
                        pad_cval=255
                    ),
                    iaa.Affine(
                        shear=(-3, 3),  # shear by -16 to +16 degrees
                    ),
                    iaa.Affine(
                        rotate=(-4, 4),  # rotate by -45 to +45 degrees
                    )
                ]),

            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug


def load_visual_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    args=None,
):
    """
    Loads the visual dataset.

    """

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path))

        path = prefix + src
        dictionary = src_dict
        print('...loading imageDataset', path, len(dictionary))
        # TODO: --image-augment
        transform = transforms.Compose([
            # ImageAug(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        img_dataset = ImageDataset(path, dictionary, append_eos=True, reverse_order=False,
                                   transform=transform,
                                   font_size=args.image_font_size,
                                   image_type=args.image_type,
                                   image_font_path=args.image_font_path,
                                   image_height=args.image_height,
                                   image_width=args.image_width,
                                   image_pad_right=args.image_pad_right,
                                   image_pretrain_path=args.image_pretrain_path,
                                   image_verbose=args.image_verbose,
                                   image_use_cache=args.image_use_cache,
                                   image_samples_path=args.image_samples_path,
                               )
        src_datasets.append(
            img_dataset
        )

        tgt_datasets.append(
            data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path,
                                                 split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl)

    print('...loading imagePairDataset', split)

    return ImagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=args.left_pad_source,
        left_pad_target=args.left_pad_target,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
        image_type=args.image_type
    )


@register_task('visualmt')
class VisualMTTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # Image dataset loader parameters
        parser.add_argument('--image-type', type=str, choices=["word", "line"], default=None,
                            help='use word or line image dataset (None | word | line)')
        parser.add_argument('--image-font-path', default=None, type=str,
                            help='Font path')
        parser.add_argument('--image-font-size', default=8, type=float,
                            help='Font size. Default: %(default)s.')
        parser.add_argument('--image-width', default=64, type=int,
                            help='Image width')
        parser.add_argument('--image-height', default=32, type=int,
                            help='Image height')
        parser.add_argument('--image-pad-right', default=5, type=int,
                            help="Pixels to pad between words for '--image-type line'")
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Image Samples path')
        parser.add_argument("--image-use-cache", action='store_true',
                            help='Cache word images after generating the first time')
        parser.add_argument("--image-augment", action='store_true',
                            help='Apply image noisification for robustness purposes')
        parser.add_argument('--image-src-loss-scale', type=float, default=1.0,
                            help='Image src loss scale')
        parser.add_argument('--image-tgt-loss-scale', type=float, default=1.0,
                            help='Image tgt loss scale')
        parser.add_argument("--image-embed-type", default='concat', type=str,
                            help='Image embed type [concat, ignore]')
        parser.add_argument('--image-embed-dim', default=512, type=int,
                            help='Image embed dim')
        parser.add_argument("--image-embed-path", type=str,
                            help='Load pretrained image embeddings')
        parser.add_argument("--image-verbose", action='store_true',
                            help='Display verbose debug')
        parser.add_argument("--image-disable", action='store_true',
                            help='Disable visual')
        parser.add_argument("--image-vista-kernel-size", type=int, default=2,
                            help="kernel size for fractional max 2d pooling")
        parser.add_argument("--image-vista-width", type=float, default=0.7,
                            help="fractional max pool 2d image width (applied twice)")
        parser.add_argument("--image-pool", action='store_true',
                            help='Vista image pool')

        parser.add_argument("--image-pretrain-path", type=str, default=None,
                            help='Load pretrain sentence embeddings')

        parser.add_argument("--image-freeze-encoder-embed", action='store_true',
                            help='Freeze preloaded visual embed')

        parser.add_argument('--image-backbone', default='vista',
                            help='CNN backbone architecture. (default: vista,  others: resnet18, resnet50)')
        parser.add_argument('--image-use-bridge', action='store_true',
                            help='Creates a linear bridge from OCR (height, channels) to MT model size')
        parser.add_argument('--image-layer', default='avgpool', type=str,
                            help='If using backbone ResNet: layer [avgpool, layer4, fc]')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning(
                '--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning(
                '--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(
            args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(
            args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_visual_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            args=self.args,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
