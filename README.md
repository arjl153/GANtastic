# GANtastic: Music Style Transfer with CycleGAN

## Abstract
GANtastic leverages CycleGAN to achieve music style transfer between different genres, specifically transforming piano arrangements from Classical to Jazz. Unlike traditional methods that rely on paired data, our approach utilizes unpaired data from "The Lakh MIDI piano dataset". This project demonstrates the potential of CycleGAN in maintaining the structure of the original music while incorporating stylistic elements of the target genre.

## Introduction
Traditional methods depend heavily on paired datasets, which are often scarce in the music domain. To address this, we employ CycleGAN, which uses cycle consistency loss to perform effective style transfer with unpaired data.


## Dataset and Preprocessing
Uses "The Lakh MIDI Dataset" and preprocess it to fit our requirements:
1. Isolate the piano channel from MIDI files.
2. Split the MIDI inputs into 30-second chunks.
3. Convert the MIDI segments into piano rolls and store them as numpy arrays.

The dataset includes:
- 12,341 Jazz samples
- 16,545 Classical samples
- 20,780 Pop samples

## Model Architecture
Implemented a CycleGAN model with two generators and two discriminators. The generators and discriminators are trained with the following loss functions:
1. Adversarial Loss
2. Identity Loss
3. Forward Cycle Loss
4. Backward Cycle Loss

The discriminator architecture includes convolutional blocks and a dense layer, while the generator architecture comprises convolutional layers followed by activation functions and batch normalization.

## Results and Conclusion
The model successfully performs symbolic music style transfer, generating realistic and pleasing music samples. The generated outputs maintain the overall structure of the original music while incorporating stylistic changes of the target genre. The best results were observed around 200 epochs of training.

## Future Work
To enhance the effectiveness of symbolic music style transfer, future work could explore:
- Leveraging ResNext blocks
- Adopting multi-class classifiers for discriminators
- Implementing complex GAN architectures like ReCycle GAN and BiCycle GAN
