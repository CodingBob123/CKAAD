# Enhanced Discriminator for CKAAD

This repository contains enhanced discriminator implementations for the CKAAD model, incorporating multi-head attention, spectral normalization, and position encoding to improve anomaly detection performance.

## Key Features

- **Multi-head Attention**: Enables the discriminator to capture spatial relationships between features
- **Spectral Normalization**: Stabilizes training by constraining the Lipschitz constant of the discriminator
- **Position Encoding**: Adds spatial awareness to the discriminator
- **Support for both Image-level and Patch-level Discriminators**: Flexible architecture for different anomaly detection tasks

## Implementation Details

We've implemented two types of discriminators:

1. **Image-level Discriminator**: Outputs a single scalar energy value for the entire image
2. **Patch-level Discriminator**: Outputs a score map with per-patch energy values

Both discriminators can be enhanced with:
- Multi-head attention for better feature representation
- Spectral normalization for training stability
- Position encoding for spatial awareness

## Usage

To use the enhanced discriminators, run `main_enhanced.py` instead of the original `main.py`:

```bash
python main_enhanced.py --discriminator_mode image --use_attention --use_spectral_norm --use_position_encoding
```

### Command-line Arguments

- `--discriminator_mode`: Choose between 'image' (image-level) or 'patch' (patch-level) discriminator
- `--use_spectral_norm`: Enable spectral normalization (default: True)
- `--use_attention`: Enable multi-head attention (default: True)
- `--use_position_encoding`: Enable position encoding (default: True)
- `--margin`: Margin for hinge loss in patch-level discriminator (default: 5.0)

## Examples

### Image-level Discriminator with Attention

```bash
python main_enhanced.py --discriminator_mode image --use_attention --use_spectral_norm --use_position_encoding
```

### Patch-level Discriminator with Attention

```bash
python main_enhanced.py --discriminator_mode patch --use_attention --use_spectral_norm --use_position_encoding --margin 3.0
```

### Image-level Discriminator without Attention

```bash
python main_enhanced.py --discriminator_mode image --use_spectral_norm --use_position_encoding
```

## Theoretical Background

The enhanced discriminators maintain compatibility with the original CKAAD theoretical framework:

- For image-level discriminator: Implements equations (8) and (9) from the paper
- For patch-level discriminator: Implements equations (11) and (12) from the paper

The modifications only enhance the function class of the discriminator without changing the adversarial game formulation, preserving the theoretical guarantees of the original method.

## Implementation Notes

- The image-level discriminator uses a pyramid structure to process multi-scale features
- The patch-level discriminator processes each scale independently with 1×1 convolutions
- Both discriminators ensure non-negative energy outputs using Softplus activation
- Gradient penalty is applied for both discriminator types to enforce Lipschitz constraint
- Position encoding is added to the input features before processing 