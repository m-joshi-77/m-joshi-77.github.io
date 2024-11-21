## Stable Diffusion: Discussion and Comparison with GAN 

### Introduction  
While GANs have been the backbone of generative AI, they come with challenges like training instability, mode collapse, and balancing generator-discriminator dynamics. Enter **Stable Diffusion**—a diffusion-based generative model that overcomes many GAN limitations by employing probabilistic modeling and latent variable frameworks.  

In this post, I’ll discuss Stable Diffusion, explore its VAE-based architecture, and a code example while comparing it with GANs to highlight its advantages.



### What is Stable Diffusion?  

Stable Diffusion leverages **diffusion probabilistic models** to iteratively transform noise into data. Instead of directly generating data as GANs do, Stable Diffusion learns a **denoising process**, making it more stable and capable of producing high-quality results with better diversity. 


### Core Component of Stable Diffusion  

#### Variational Autoencoders (VAE)  
VAEs compress high-dimensional data (like images) into a latent space and reconstruct it. Stable Diffusion operates in this latent space, making training more efficient.

#### VAE Architecture  
- **Encoder**: Compresses data into latent space.  
- **Decoder**: Reconstructs data from latent representations.

#### Core Code Example: VAE  
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # Outputs twice the latent dimension (latent_dim * 2) to compute: Latent mean and Sampling Log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # Sigmoid to normalize outputs. 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Ensures differentiability for gradient-based optimization.

    def forward(self, x):
        x = x.view(-1, 784)
        stats = self.encoder(x)
        mu, logvar = stats[:, :latent_dim], stats[:, latent_dim:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```
## How the Diffusion Process Operates in Latent Space  

In the VAE core code, the **latent space** is represented by the variable `z`, which is computed using the **reparameterize** method. The diffusion process operates in this latent space as follows:  

- The encoder compresses the high-dimensional input `x` into a compact latent representation `z`.  
- The diffusion process can then act directly on `z`, as it is a low-dimensional representation of the input data that retains essential features.  

To integrate a diffusion process, `z` should be passed to a diffusion model for further processing.

Steps to incorporate a diffusion process:  

1. **Use Latent Space**: Replace the decoder with a diffusion model to operate on `z`.  

2. **Forward Process**: Add Gaussian noise to `z` to simulate data corruption:  
   ```python
   noise = torch.randn_like(z)  # Add Gaussian noise
   x_t = z + noise  # Forward diffusion
    ```
3. Reverse Process: Train a diffusion model to denoise x_t and recover `z`:
   ```python
   pred_z = diffusion_model(x_t, t)  # Reverse diffusion
   ```
4. Decode `z`: After denoising, pass `z` to the decoder to reconstruct the original input data.

This integration allows the diffusion process to work in the compact, low-dimensional latent space created by the VAE.

[<img src="/assets/images/vaediffusion.png" alt="VAE with Diffusion Model Decoder" style="width:50%; max-width:600px;" />]: #
<img src="/assets/images/vaediffusion.png" alt="VAE with Diffusion Model Decoder" style="display:block; margin-left:auto; margin-right:auto; width:50%; max-width:600px;" />


#### Replacing the Decoder with a Diffusion Model in Core Code Example

```python
import torch
import torch.nn as nn

# Define Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, t):
        return self.model(x)

# Define VAE with Diffusion Integration
class VAEWithDiffusion(nn.Module):
    def __init__(self, latent_dim):
        super(VAEWithDiffusion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # Latent mean and variance
        )
        # Replace decoder with a diffusion model
        self.diffusion_model = DiffusionModel(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t):
        # Encode input into latent space
        x = x.view(-1, 784)
        stats = self.encoder(x)
        mu, logvar = stats[:, :latent_dim], stats[:, latent_dim:]
        z = self.reparameterize(mu, logvar)

        # Apply diffusion process to z
        noisy_z = z + torch.randn_like(z)  # Add Gaussian noise
        denoised_z = self.diffusion_model(noisy_z, t)  # Denoise latent space

        return denoised_z, mu, logvar
```

## Why Stable Diffusion is Better than GANs  

| **Feature**       | **Stable Diffusion**                  | **GANs**                           |
|--------------------|---------------------------------------|-------------------------------------|
| **Stability**      | Robust training with fewer failures  | Prone to instability                |
| **Mode Collapse**  | Avoided due to iterative denoising   | Common problem                      |
| **Output Diversity** | Generates diverse samples effectively | Limited by discriminator learning   |
| **Efficiency**     | Operates in compressed latent space  | Directly works in high dimensions- costly   |

## Applications of Stable Diffusion  

- **High Resolution and Diverse Image Generation**: Example - DALL·E 2.  
- **Inpainting**: Means filling in missing parts of an image.  
- **Enhancing the Quality of Low Resolution Images**.  

## Conclusion  

Stable Diffusion is a significant leap in generative modeling, addressing the shortcomings of GANs while unlocking new applications in high-quality image synthesis, text-to-image generation. By integrating concepts like VAEs and diffusion processes, it ensures robust, stable, and diverse outputs.

