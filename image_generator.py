from PIL import Image, ImageDraw, ImageFont, ImageStat, ImageEnhance
import torch as th
from datetime import date
import math
import os

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


class ImageGenerator:
    """
    class for generating a "WordOfTheDay" style image
    """
    def __init__(self):
        """
        Initializes the generator
        """
        batch_size = 1
        upsample_temp = 1.0
        self.text2im = Text2Im(batch_size)
        self.upsampler = Upsampler(batch_size, upsample_temp)

    def generate(self, word, pos, pronunciation, meaning):
        """
        Generates a "WordOfTheDay" style image with the given word and meaning
        Params:
            word: the word as a string
            pos: the part of speech of the word as a string
            pronunciation: the pronunciation of the word as a string
            meaning: the meaning of the word as a string
        """

        # Generate an image from the meaning
        samples = self.text2im.generate(meaning)

        # Upsample it
        up_samples = self.upsampler.generate(samples, meaning)

        # Convert to PIL.Image
        im = self._image_from_samples(up_samples)

        # Crop and resize to the correct dimensions
        im = self._crop_and_resize(im)

        # Adjust brightness
        im = self._adjust_brightness(im)

        # Apply the text to the image
        im = self._apply_text(im, word, pos, pronunciation, meaning)

        return im

    def _adjust_brightness(self, im):
        """
        Lowers the brightness of the given image if the overall brightness is too high
        Params:
            im: an image as a PIL.Image
        Returns:
            the image with adjusted brightness
        """
        stat = ImageStat.Stat(im)
        r, g, b = stat.mean
        brightness = math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
        if brightness > 150:
            enhancer = ImageEnhance.Brightness(im)
            im = enhancer.enhance(1 - (brightness-150) / 255)
        return im

    def _apply_text(self, image, word, pos, pronunciation, meaning):
        """
        Applies the text to the given image to create a #WordOfTheDay style image
        """

        draw = ImageDraw.Draw(image)
        w, h = image.size
        white = (255, 255, 255)

        # Draw logo
        logo = Image.open(os.path.join("data", "logo.png"))
        image.paste(logo, (275, 50), logo)

        # Draw header and date
        font_size = 60
        font = ImageFont.truetype(os.path.join("data", "fonts", "Lato-Regular.ttf"), font_size)
        wotd_text = "W O R D   O F   T H E   D A Y"
        date_text = " ".join(list(date.today().strftime("%B %d, %Y").upper()))
        draw.text((500, 50), wotd_text, white, font=font, anchor="la")
        draw.line((500, 140, 1220, 140), white, width=5)
        draw.text((500, 150), date_text, white, font=font, anchor="la")

        # Draw word
        font_size = 210 - max(0, len(word) - 10) * 10
        font = ImageFont.truetype(os.path.join("data", "fonts", "PlayfairDisplay-Regular.ttf"), font_size)
        draw.text((w // 2, h // 2 + 100), word, white, font=font, anchor="md")

        # Draw part of speech and pronunciation
        pos_text = " ".join(list(pos))
        pron_text = "   |   " + " ".join(list(pronunciation))

        font_size = 40
        pos_font = ImageFont.truetype(os.path.join("data", "fonts", "Lato-Italic.ttf"), font_size)
        pos_length = draw.textlength(pos_text, font=pos_font)
        pron_font = ImageFont.truetype(os.path.join("data", "fonts", "Lato-Regular.ttf"), font_size)
        pron_length = draw.textlength(pron_text, font=pron_font)
        total_length = pos_length + pron_length

        pos_start = w // 2 - total_length // 2
        pron_end = w // 2 + total_length // 2

        draw.text((pos_start, h // 2 + 100), pos_text, white, font=pos_font, anchor="la")
        draw.text((pron_end, h // 2 + 100), pron_text, white, font=pron_font, anchor="ra")

        # Draw meaning
        meaning_text = " ".join(list(meaning))

        font_size = 60
        font = ImageFont.truetype(os.path.join("data", "fonts", "Lato-Bold.ttf"), font_size)

        text_length = draw.textlength(meaning_text, font=font)
        if text_length > 2 * w - 80:
            font_size = 45
            font = ImageFont.truetype(os.path.join("data", "fonts", "Lato-Bold.ttf"), font_size)
            split_words = meaning.split()
            third = len(split_words) // 3
            meaning_text = " ".join(split_words[:third]) + "\n" + " ".join(
                split_words[third:2 * third]) + "\n" + " ".join(split_words[2 * third:])
            meaning_text = " ".join(list(meaning_text))
            draw.multiline_text((w // 2, h - 60), meaning_text, white, font=font, anchor="md", spacing=12,
                                align="center")
        elif text_length > w - 40:
            split_words = meaning.split()
            middle = len(split_words) // 2
            meaning_text = " ".join(split_words[:middle]) + "\n" + " ".join(split_words[middle:])
            meaning_text = " ".join(list(meaning_text))
            draw.multiline_text((w // 2, h - 80), meaning_text, white, font=font, anchor="md", spacing=8,
                                align="center")
        else:
            draw.text((w // 2, h - 80), meaning_text, white, font=font, anchor="md")

        return image

    def _image_from_samples(self, batch):
        """
        Create an image from a torch.Tensor
        Params:
            batch: the torch.Tensor representing the image
        Returns:
            a PIL.Image object
        """
        scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        return Image.fromarray(reshaped.numpy())

    def _crop_and_resize(self, image):
        """
        Crops and resizes the given image to a 1500x800 format
        Params:
            image: the image as a PIL.Image
        Returns:
            image: the cropped and resized image
        """
        width, height = image.size
        new_width, new_height = width, width * 820//1500
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        image = image.crop((left, top, right, bottom))

        image = image.resize((1500, 820), Image.LANCZOS)
        return image

    def save_image(self, word, image):
        """
        Saves the images in a folder called image_samples
        Params:
            word: the word of the day on the image
            image: the image as a PIL.Image
        """
        image.save(os.path.join("image_samples", f"{word}_{image.size[0]}x{image.size[1]}.png"))


class Text2Im:
    """
    class for generating an image from a text string
    Adapted from https://github.com/openai/glide-text2im
    """
    def __init__(self, batch_size):
        """
        Initializes the text to image generator
        Params:
            batch_size: the batch_size to be used by the model
        """

        has_cuda = th.cuda.is_available()
        self.device = th.device('cpu' if not has_cuda else 'cuda')

        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = has_cuda
        self.options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling

        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.model.eval()
        if has_cuda:
            self.model.convert_to_fp16()
        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint('base', self.device))

        print('total base parameters', sum(x.numel() for x in self.model.parameters()))

        # Sampling parameters
        self.batch_size = batch_size
        self.guidance_scale = 3.0

    def generate(self, prompt):
        """
        Samples an image from the model with a given prompt
        Params:
            prompt: the prompt as a string
        Returns:
            the image represented as a torch.Tensor
        """

        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = self.batch_size * 2
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * self.batch_size + [uncond_tokens] * self.batch_size, device=self.device
            ),
            mask=th.tensor(
                [mask] * self.batch_size + [uncond_mask] * self.batch_size,
                dtype=th.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            self._model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model.del_cache()

        return samples

    def _model_fn(self, x_t, ts, **kwargs):
        """
        Create a classifier-free guidance sampling function
        """
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)


class Upsampler:
    """
    class for upsampling an image to 256x256
    Adapted from https://github.com/openai/glide-text2im
    """
    def __init__(self, batch_size, upsample_temp=0.997):
        """
        Initializes the upsampler model
        Params:
            batch_size: the batch_size to be used by the model
            upsample_temp: the upsampling temperature
        """
        has_cuda = th.cuda.is_available()
        self.device = th.device('cpu' if not has_cuda else 'cuda')

        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up['use_fp16'] = has_cuda
        self.options_up['timestep_respacing'] = 'fast27'  # use 27 diffusion steps for very fast sampling
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)
        self.model_up.eval()
        if has_cuda:
            self.model_up.convert_to_fp16()
        self.model_up.to(self.device)
        self.model_up.load_state_dict(load_checkpoint('upsample', self.device))
        print('total upsampler parameters', sum(x.numel() for x in self.model_up.parameters()))

        # Sampling parameters
        self.batch_size = batch_size
        self.upsample_temp = upsample_temp

    def generate(self, samples, prompt):
        """
        Upsamples the image
        Params:
            samples: the image as a torch.Tensor
            prompt: the prompt used to generate the image
        Returns:
            the upsampled image as a torch.Tensor
        """

        tokens = self.model_up.tokenizer.encode(prompt)
        tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
            tokens, self.options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * self.batch_size, device=self.device
            ),
            mask=th.tensor(
                [mask] * self.batch_size,
                dtype=th.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.model_up.del_cache()
        up_shape = (self.batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.diffusion_up.ddim_sample_loop(
            self.model_up,
            up_shape,
            noise=th.randn(up_shape, device=self.device) * self.upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model_up.del_cache()

        return up_samples

